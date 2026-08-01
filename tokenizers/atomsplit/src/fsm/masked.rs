//! Boundary-mask scanners: the SIMD replacement for the per-run scalar walk of the regex-shaped
//! FSMs (one scheme module per regex family, sharing this walker).
//!
//! A scalar FSM advances one token at a time: a rule dispatch, then a per-byte run scan. A
//! masked scanner instead classifies 64 bytes at once into per-class bitmasks (bit k set =
//! "byte k is a letter") and computes every token start in the batch with a few dozen
//! branch-free u64 operations. That works because these regexes are class-run languages with
//! one or two chars of context: a token starts exactly at a class change that is not an
//! absorbed prefix, at a whitespace-run edge, or at a contraction edge, and each of those
//! conditions is a shifted-mask expression. The boundary algebra is transcribed per scheme from
//! gigatoken's `src/pretokenize/fast/` scanners (MIT), with one structural difference:
//! gigatoken classifies raw bytes in-batch and falls back to a per-char loop whenever a batch
//! contains a non-ASCII byte, while these scanners build their masks from the
//! [`crate::classify`] tag stream, which is already SIMD and covers all of Unicode. A
//! continuation byte's tag is [`Atom::Cont`]; the fill step gives it its char's class, so byte
//! adjacency equals char adjacency and the same algebra applies to non-ASCII batches.
//!
//! # Trust boundaries
//!
//! Each scheme's `batch_masks` returns `(boundary, bad)` for one 64-byte batch. A `boundary`
//! bit is a proven token start. A `bad` bit means the algebra cannot decide that byte (batch-
//! edge straddles, char-counted rules over multi-byte chars, run-contextual classes; each
//! scheme documents its own list). `boundary & bad` is always 0, and no span is emitted across
//! a bad zone: the walker re-derives tokens there with the scheme's scalar `advance`, the same
//! rules the plain scan runs, and resumes on masks at the next batch. The scalar scans stay the
//! ground truth; the `masked_*` tests (tests/fsm.rs) pin byte-exactness at every batch-edge
//! offset.
//!
//! # Targets
//!
//! The per-target work is confined to [`block`] (64-byte predicate masks): aarch64 NEON,
//! x86_64 SSE2 and wasm32 simd128 have kernels; every other target delegates the
//! `scan_*_masked` entry points to the scalar scans, so correctness never depends on a SIMD
//! path being present.
//!
//! Inputs must be well-formed UTF-8 (the crate-level contract); the fill step relies on
//! continuation runs of at most 3 bytes.

use super::*;

#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
mod block;
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
mod byte_level;
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
mod cl100k;
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
mod deepseek;
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
mod o200k;

/// [`fsm_byte_level`] over the masked scanner: writes spans into `out` (len >= `text.len()`)
/// and returns the count.
#[must_use]
pub fn fsm_byte_level_masked(text: &[u8], tags: &[u8], out: &mut [Span]) -> usize {
    debug_assert!(out.len() >= text.len() && tags.len() >= text.len());
    let mut w = 0usize;
    scan_byte_level_masked(text, tags, |span| {
        // SAFETY: tokens partition the input, so `w < #tokens <= text.len() <= out.len()`.
        unsafe { *out.get_unchecked_mut(w) = span };
        w += 1;
    });
    w
}

/// The masked twin of [`scan_byte_level`]: same tokens, same emit order. Targets without a
/// [`block`] kernel delegate to the scalar scan.
pub fn scan_byte_level_masked(text: &[u8], tags: &[u8], emit: impl FnMut(Span)) {
    #[cfg(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    walk(&byte_level::ByteLevelMasked, text, tags, emit);
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    scan_byte_level(text, tags, emit);
}

/// [`fsm_cl100k_cap`] over the masked scanner: writes spans into `out` (len >= `text.len()`)
/// and returns the count.
#[must_use]
pub fn fsm_cl100k_cap_masked(
    text: &[u8],
    tags: &[u8],
    out: &mut [Span],
    digit_cap: usize,
) -> usize {
    debug_assert!(out.len() >= text.len() && tags.len() >= text.len());
    let mut w = 0usize;
    scan_cl100k_cap_masked(text, tags, digit_cap, |span| {
        // SAFETY: tokens partition the input, so `w < #tokens <= text.len() <= out.len()`.
        unsafe { *out.get_unchecked_mut(w) = span };
        w += 1;
    });
    w
}

/// The masked twin of [`scan_cl100k_cap`]: same tokens, same emit order. Digit caps other than
/// 1, 3 and `usize::MAX` (none ship today) and targets without a [`block`] kernel delegate to
/// the scalar scan.
pub fn scan_cl100k_cap_masked(text: &[u8], tags: &[u8], digit_cap: usize, emit: impl FnMut(Span)) {
    #[cfg(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    {
        if matches!(digit_cap, 1 | 3 | usize::MAX) {
            walk(&cl100k::Cl100kMasked { digit_cap }, text, tags, emit);
        } else {
            scan_cl100k_cap(text, tags, digit_cap, emit);
        }
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    scan_cl100k_cap(text, tags, digit_cap, emit);
}

/// [`fsm_o200k`] over the masked scanner: writes spans into `out` (len >= `text.len()`) and
/// returns the count.
#[must_use]
pub fn fsm_o200k_masked(text: &[u8], tags: &[u8], out: &mut [Span]) -> usize {
    debug_assert!(out.len() >= text.len() && tags.len() >= text.len());
    let mut w = 0usize;
    scan_o200k_masked(text, tags, |span| {
        // SAFETY: tokens partition the input, so `w < #tokens <= text.len() <= out.len()`.
        unsafe { *out.get_unchecked_mut(w) = span };
        w += 1;
    });
    w
}

/// [`fsm_tekken`] over the masked scanner: writes spans into `out` (len >= `text.len()`) and
/// returns the count.
#[must_use]
pub fn fsm_tekken_masked(text: &[u8], tags: &[u8], out: &mut [Span]) -> usize {
    debug_assert!(out.len() >= text.len() && tags.len() >= text.len());
    let mut w = 0usize;
    scan_tekken_masked(text, tags, |span| {
        // SAFETY: tokens partition the input, so `w < #tokens <= text.len() <= out.len()`.
        unsafe { *out.get_unchecked_mut(w) = span };
        w += 1;
    });
    w
}

/// The masked twin of [`scan_o200k`]: same tokens, same emit order. Targets without a
/// [`block`] kernel delegate to the scalar scan.
pub fn scan_o200k_masked(text: &[u8], tags: &[u8], emit: impl FnMut(Span)) {
    #[cfg(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    walk(&o200k::O200kMasked::<true, 3>, text, tags, emit);
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    scan_o200k(text, tags, emit);
}

/// The masked twin of [`scan_tekken`]: same tokens, same emit order. Targets without a
/// [`block`] kernel delegate to the scalar scan.
pub fn scan_tekken_masked(text: &[u8], tags: &[u8], emit: impl FnMut(Span)) {
    #[cfg(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    walk(&o200k::O200kMasked::<false, 1>, text, tags, emit);
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    scan_tekken(text, tags, emit);
}

/// [`fsm_deepseek`] over the masked scanner: writes spans into `out` (len >= `text.len()`) and
/// returns the count.
#[must_use]
pub fn fsm_deepseek_masked(text: &[u8], tags: &[u8], out: &mut [Span]) -> usize {
    debug_assert!(out.len() >= text.len() && tags.len() >= text.len());
    let mut w = 0usize;
    scan_deepseek_masked(text, tags, |span| {
        // SAFETY: tokens partition the input, so `w < #tokens <= text.len() <= out.len()`.
        unsafe { *out.get_unchecked_mut(w) = span };
        w += 1;
    });
    w
}

/// The masked twin of [`scan_deepseek`]: same tokens, same emit order. Targets without a
/// [`block`] kernel delegate to the scalar scan.
pub fn scan_deepseek_masked(text: &[u8], tags: &[u8], emit: impl FnMut(Span)) {
    #[cfg(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    walk(&deepseek::DeepSeekMasked, text, tags, emit);
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    scan_deepseek(text, tags, emit);
}

/// One masked scheme: the batch classifier and the scalar rules the walker falls back on.
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
trait MaskedFsm {
    /// `(boundary, bad)` for `text[scan..scan + 64]`: `boundary` bit k = proven token start at
    /// `scan + k`, `bad` bit k = the walker must re-derive byte `scan + k` with
    /// [`Self::advance`]. `boundary & bad` must be 0. Callers guarantee `scan + 64 <
    /// text.len()` (one lookahead tag is readable).
    fn batch_masks(&self, text: &[u8], tags: &[u8], scan: usize) -> (u64, u64);

    /// Scalar ground truth: end of the token starting at `i` (`i < end`, `i` on a token
    /// boundary).
    fn advance(&self, text: &[u8], tags: &[u8], i: usize, end: usize) -> usize;
}

/// The batch walker: consume proven token starts batch by batch, re-derive bad zones with the
/// scheme's scalar rules. `pending` is always the start of the open (not yet emitted) token.
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
fn walk(scheme: &impl MaskedFsm, text: &[u8], tags: &[u8], mut emit: impl FnMut(Span)) {
    let end = text.len();
    let tags = &tags[..end];
    let mut pending = 0usize;
    let mut scan = 0usize;
    // Each batch reads the lookahead tag at scan + 64 (the `\s+(?!\S)` bit-63 rules), so the
    // last <= 64 bytes always go through the scalar tail below.
    while scan + 64 < end {
        if scan + 64 <= pending {
            // A scalar re-derivation ran past this whole batch.
            scan += 64;
            continue;
        }
        let (boundary, mut bad) = scheme.batch_masks(text, tags, scan);
        let mut m = boundary;
        if pending > scan {
            // Bits at or below `pending` are inside or at the start of the open token; they are
            // stale leftovers of a scalar re-derivation that entered this batch.
            let done = pending - scan + 1;
            m = if done >= 64 {
                0
            } else {
                m & (u64::MAX << done)
            };
        }
        // Interleave: consume proven starts below the next bad zone, re-derive the zone with
        // the scalar rules, repeat. A span must never be emitted across an unresolved zone, so
        // starts above one cannot pair with `pending` from below it.
        loop {
            let zone = if bad == 0 {
                64
            } else {
                bad.trailing_zeros() as usize
            };
            while m != 0 {
                let j = m.trailing_zeros() as usize;
                if j >= zone {
                    break;
                }
                let p = scan + j;
                if p > pending {
                    emit(Span {
                        start: pending as u32,
                        end: p as u32,
                    });
                    pending = p;
                }
                m &= m - 1;
            }
            if bad == 0 {
                break;
            }
            // The zone's contiguous extent; the scalar rules resolve through its end (their
            // tokens may overshoot it, or the whole batch — later bits fall to the guards).
            let zone_end = zone + ((!(bad >> zone)).trailing_zeros() as usize).min(64 - zone);
            while pending < scan + zone_end {
                let e = scheme.advance(text, tags, pending, end);
                emit(Span {
                    start: pending as u32,
                    end: e as u32,
                });
                pending = e;
            }
            bad = if zone_end >= 64 {
                0
            } else {
                bad & (u64::MAX << zone_end)
            };
        }
        scan += 64;
    }
    while pending < end {
        let e = scheme.advance(text, tags, pending, end);
        emit(Span {
            start: pending as u32,
            end: e as u32,
        });
        pending = e;
    }
}

// ── shared u64 helpers (platform-independent; the scheme modules compose these) ────────────────

/// The two continuation-run masks the fill steps need: `c2` bit k = bytes k and k-1 are both
/// continuations, `c3` = bytes k, k-1, k-2 all are.
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
fn cont_runs(c: u64) -> (u64, u64) {
    let c2 = c & (c << 1);
    (c2, c2 & (c << 2))
}

/// Fill: every continuation byte of a char whose lead is in `m` joins `m`, so byte adjacency
/// equals char adjacency (UTF-8 chars are at most 4 bytes: 3 hops cover every continuation).
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
fn fill(m: u64, c: u64, c2: u64, c3: u64) -> u64 {
    m | ((m << 1) & c) | ((m << 2) & c2) | ((m << 3) & c3)
}

/// Smear `seed` upward (toward higher bits) through contiguous set bits of `within`, in log
/// steps (via gigatoken's `cl100k_family.rs`, MIT).
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
fn smear_up(seed: u64, within: u64) -> u64 {
    let mut a = seed;
    let mut m = within;
    let mut sh = 1u32;
    while sh < 64 {
        a |= (a << sh) & m;
        m &= m << sh;
        sh <<= 1;
    }
    a
}

/// Token-start bits inside ASCII digit runs for `\p{N}{1,3}`: each run splits into 3-char
/// tokens, so boundaries sit at run start + 3k (via gigatoken's `mask.rs`, MIT). Callers keep
/// multi-byte digit chars out of `d`: their grouping is char-counted, and byte hops would
/// misphase it.
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
fn digit_run_splits3(d: u64) -> u64 {
    let mut b = d & !(d << 1); // run starts
    // A start at p re-arms at p+3 while the run continues: hop condition c = "p..p+3 all
    // digits". Log-doubling covers 64-bit runs in 5 steps.
    let mut c = d & (d >> 1) & (d >> 2) & (d >> 3);
    let mut sh = 3u32;
    while sh < 64 {
        b |= (b & c) << sh;
        c &= c >> sh;
        sh <<= 1;
    }
    b
}

/// `x << n`, saturating to 0 at `n >= 64` (`trailing_zeros` on an empty mask yields 64).
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
fn shl_sat(x: u64, n: u32) -> u64 {
    if n >= 64 { 0 } else { x << n }
}

/// The lead of the char containing `tags[p]`: walk back over continuation tags (at most 3 on
/// well-formed UTF-8; `p` itself may already be the lead).
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
fn char_lead(tags: &[u8], mut p: usize) -> usize {
    while p > 0 && tags[p] & 0x0F == CONT {
        p -= 1;
    }
    p
}
