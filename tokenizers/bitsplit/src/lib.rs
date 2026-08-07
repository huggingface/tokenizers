//! `bitsplit` — GPT-family pre-tokenization as a **bitstream program**, replacing the scalar FSMs.
//!
//! Follows *Interleaved Bitstream Execution for Multi-Pattern Regex Matching on GPUs*
//! (MICRO'25, doi 10.1145/3725843.3756052). The paper's two ideas that carry over to a CPU:
//!
//!   1. **Bit-parallel regex.** Compile the grammar into character-class *bitstreams* (one bit per
//!      input byte) plus boolean ops and carry-propagating adds. 64 input bytes are decided per
//!      64-bit register op, branchlessly — the FSM's per-token unpredictable branch disappears.
//!   2. **Interleaved execution.** Do NOT run one loop per bitstream instruction over the whole
//!      input (that is the "sequential" baseline the paper beats: every intermediate stream is
//!      materialised and re-read). Instead fuse *all* instructions into ONE block-wise loop, so an
//!      intermediate stream lives in a register for the ~100 ops it is needed and dies there. The
//!      only thing that reaches memory is the final `starts` bitmap (n/8 bytes).
//!
//! The paper's third contribution (dependency-aware thread-data mapping) is a GPU concern — it
//! resolves cross-block dependencies by recomputing them on other SMs. On one core the blocks are
//! visited in order, so those dependencies are carried in a handful of scalar registers and, for
//! the one genuinely backward-in-time rule, patched into the already-written bitmap.
//!
//! We do not re-derive character classes: [`classify`] already emits one `Atom` tag per byte. The
//! builder folds those 16 atoms into a grammar-specific **dense 3-bit code** and extracts 3
//! bit-planes of it; every class stream is then a 2–3 op boolean function of the planes.
//!
//! Grammars: [`bitsplit_deepseek`], [`bitsplit_byte_level`] (GPT-2), [`bitsplit_cl100k`]. All three
//! byte-exact with the oniguruma oracle over a block-phase sweep — see `tests/parity.rs`.

mod atom_tables;
pub mod classify;
pub mod deepseek;
pub mod gpt;
pub mod literal;
pub mod regexes;
#[cfg(target_arch = "aarch64")]
mod simd;
#[cfg(target_arch = "aarch64")]
mod simd_classify;
#[cfg(target_arch = "x86_64")]
mod simd_avx_classify;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod simd_wasm_classify;
#[cfg(target_arch = "x86_64")]
mod simd_x86;
mod tables;

pub use deepseek::bitsplit_deepseek;
pub use gpt::{bitsplit_byte_level, bitsplit_cl100k};

/// A token span: byte offsets `[start, end)` into the input. `#[repr(C)]` so the output buffer has a
/// stable `[start, end]` layout — the pipeline reuses it with zero conversion, and it can be
/// reinterpreted as bytes / handed across the crate boundary.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash, PartialOrd, Ord)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

impl Span {
    #[inline]
    #[must_use]
    pub const fn new(start: u32, end: u32) -> Self {
        Self { start, end }
    }

    /// `[start, end)` as a `usize` range — for slicing the input text.
    #[inline]
    #[must_use]
    pub fn range(self) -> core::ops::Range<usize> {
        self.start as usize..self.end as usize
    }
}

impl PartialEq<(u32, u32)> for Span {
    fn eq(&self, other: &(u32, u32)) -> bool {
        self.start == other.0 && self.end == other.1
    }
}

/// Whether this target builds its bitstreams with SIMD. Without it the builder is the portable
/// byte-at-a-time reference, which is slower than the FSM this replaces -- so a caller should keep
/// its FSM path rather than route here. On x86 the kernel needs SSSE3, so this is a runtime check.
#[must_use]
pub fn fast_builder() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        true
    }
    #[cfg(target_arch = "x86_64")]
    {
        has_ssse3()
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
fn has_ssse3() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHED: AtomicU8 = AtomicU8::new(0);
    match CACHED.load(Ordering::Relaxed) {
        0 => {
            let yes = std::arch::is_x86_feature_detected!("ssse3");
            CACHED.store(1 + u8::from(yes), Ordering::Relaxed);
            yes
        }
        n => n == 2,
    }
}

pub(crate) const CONT: u8 = 15; // Atom::Cont
pub(crate) const CODE_CONT: u8 = 7; // every grammar's dense code for a continuation byte

/// The bitstreams for one 64-byte block. `p0`/`p1`/`p2` are the bit-planes of the **filled** dense
/// code — filled meaning a multi-byte char sets its bits on *all* of its bytes, so "previous char's
/// class" is a plain `<< 1` and no rule does char-width arithmetic. `cont` is the one un-filled
/// stream (it defines the fill); `cjk` is text-derived and only built when a grammar asks for it.
#[derive(Default, Clone, Copy)]
pub(crate) struct Blk {
    pub cont: u64,
    pub p0: u64,
    pub p1: u64,
    pub p2: u64,
    pub cjk: u64,
}

/// deepseek Split-2's isolated range: Han U+4E00..9FA5 ∪ Hiragana/Katakana U+3040..30FF (all
/// 3-byte, leads E3..E9). Same predicate as `fsm_deepseek`'s `ds_is_cjk_at`.
#[inline]
pub(crate) fn is_cjk_at(text: &[u8], p: usize) -> bool {
    let b = text[p];
    if !(0xE3..=0xE9).contains(&b) || p + 2 >= text.len() {
        return false;
    }
    let cp = ((b as u32 & 0x0F) << 12)
        | ((text[p + 1] as u32 & 0x3F) << 6)
        | (text[p + 2] as u32 & 0x3F);
    (0x4E00..=0x9FA5).contains(&cp) || (0x3040..=0x30FF).contains(&cp)
}

/// Build one block. Full blocks go through the NEON kernel; the ragged tail (and every other
/// target) uses the portable byte-at-a-time reference. Both produce the identical `Blk`.
/// `CJK` asks for the (deepseek-only) range stream; `lut` is the grammar's tag → dense code table.
#[inline]
pub(crate) fn build_block<const CJK: bool>(
    text: &[u8],
    tags: &[u8],
    base: usize,
    len: usize,
    lut: &[u8; 64],
    cur_code: u8,
    cur_cjk: bool,
) -> (Blk, u8) {
    #[cfg(target_arch = "aarch64")]
    if len == 64 {
        // SAFETY: `len == 64` means `base + 64 <= text.len() == tags.len()`.
        return unsafe { crate::simd::build64::<CJK>(text, tags, base, lut, cur_code, cur_cjk) };
    }
    #[cfg(target_arch = "x86_64")]
    if len == 64 && has_ssse3() {
        // SAFETY: `len == 64` bounds both reads, and SSSE3 is checked above.
        return unsafe {
            crate::simd_x86::build64::<CJK>(text, tags, base, lut, cur_code, cur_cjk)
        };
    }
    build_block_scalar::<CJK>(text, tags, base, len, lut, cur_code, cur_cjk)
}

/// Portable reference builder: one byte at a time.
pub(crate) fn build_block_scalar<const CJK: bool>(
    text: &[u8],
    tags: &[u8],
    base: usize,
    len: usize,
    lut: &[u8; 64],
    mut cur_code: u8,
    mut cur_cjk: bool,
) -> (Blk, u8) {
    let mut b = Blk::default();
    for i in 0..len {
        let p = base + i;
        let bit = 1u64 << i;
        if tags[p] == CONT {
            b.cont |= bit;
        } else {
            cur_code = lut[tags[p] as usize];
            cur_cjk = CJK && is_cjk_at(text, p);
        }
        b.p0 |= bit * u64::from(cur_code & 1 != 0);
        b.p1 |= bit * u64::from(cur_code & 2 != 0);
        b.p2 |= bit * u64::from(cur_code & 4 != 0);
        b.cjk |= bit * u64::from(cur_cjk);
    }
    (b, cur_code)
}

// ── bitstream primitives ────────────────────────────────────────────────────────────────────────

/// `ScanThru`: move every marker in `m` forward past the run of `c` it sits in, landing on the
/// first position not in `c`. The classic Parabix carry-propagation trick — one add. Done in `u128`
/// so a run reaching bit 63 puts its landing bit at 64 instead of vanishing: that bit *is* the
/// block's carry-out, and `e - m` still yields the right in-block span.
#[inline]
pub(crate) const fn scanthru(m: u128, c: u128) -> u128 {
    m.wrapping_add(c) & !c
}

/// Advance markers by ONE CHAR: shift one byte, then scan through the continuation bytes. Markers
/// that leave the block are dropped — their state is reconstructed from the scalar carry instead.
#[inline]
pub(crate) const fn adv(m: u64, cont: u64) -> u64 {
    scanthru((m as u128) << 1, cont as u128) as u64
}

/// Move each bit back to the lead byte of its char. Rules that look at the *next* char are stated
/// at a char's last byte; token starts must sit on leads. ≤3 steps (UTF-8 chars are ≤4 bytes).
/// Returns `(in-block, patch for the previous word)` — a char straddling the block boundary lands
/// its lead in the word we already wrote.
#[inline]
pub(crate) fn to_lead(x: u64, cont: u64, prev_cont: u64) -> (u64, u64) {
    let c = ((cont as u128) << 64) | prev_cont as u128;
    let mut y = (x as u128) << 64;
    for _ in 0..3 {
        let m = y & c;
        if m == 0 {
            break;
        }
        y = (y & !c) | (m >> 1);
    }
    ((y >> 64) as u64, y as u64)
}

/// In each run of `c`, fill from the run start through the LAST marker of `m` (`m ⊆ c`). Used on
/// reversed streams, where it answers "is there a newline at-or-after me, still inside this
/// whitespace run?". `(end - m) | m` rather than `end - m`: the latter only spans from the *first*
/// marker when a run holds several.
#[inline]
pub(crate) fn fill_to_last(m: u64, c: u64) -> u64 {
    if m == 0 {
        return 0;
    }
    let (m128, c128) = (m as u128, c as u128);
    (scanthru(m128, c128).wrapping_sub(m128) | m128) as u64
}

/// Run of `x` that starts at bit 0.
#[inline]
pub(crate) fn lead_run(x: u64, valid: u64) -> u64 {
    let z = !x & valid;
    if z == 0 {
        valid
    } else {
        (z & z.wrapping_neg()) - 1
    }
}

/// Run of `x` that ends at bit `len - 1` (0 if that bit is clear).
#[inline]
pub(crate) fn trail_run(x: u64, valid: u64, len: usize) -> u64 {
    if x & (1u64 << (len - 1)) == 0 {
        return 0;
    }
    let z = !x & valid;
    if z == 0 {
        valid
    } else {
        valid & !((1u64 << (64 - z.leading_zeros())) - 1)
    }
}

// ── emit ────────────────────────────────────────────────────────────────────────────────────────

/// `starts` bitmap → spans. Each set bit closes the previous token and opens the next; `tzcnt`
/// walks them at ~3 ops per token.
pub(crate) fn emit(starts: &[u64], nblk: usize, n: usize, out: &mut [Span]) -> usize {
    let (mut w, mut open) = (0usize, u32::MAX);
    for (bi, &word) in starts.iter().enumerate().take(nblk) {
        let mut m = word;
        while m != 0 {
            let pos = (bi * 64 + m.trailing_zeros() as usize) as u32;
            if open != u32::MAX {
                out[w] = Span::new(open, pos);
                w += 1;
            }
            open = pos;
            m &= m - 1;
        }
    }
    if open != u32::MAX {
        out[w] = Span::new(open, n as u32);
        w += 1;
    }
    w
}

/// Emit with a **scalar escape at flagged bits**: contractions (`'s 't 're 've 'm 'll 'd`) are
/// variable-length, case-optional and outrank every other alternative, which makes them miserable
/// in bit algebra and trivial here. `flag` marks the apostrophes that open a token; a block with
/// none takes the plain loop, so the escape costs one test per block on ordinary text.
///
/// A matched contraction overrides the algebra outright: it emits its own span and skips every
/// start bit inside it, which is how the letter alternative loses the tie (`'sx` → `'s`, `x`).
pub(crate) fn emit_contr(
    text: &[u8],
    starts: &[u64],
    flag: &[u64],
    nblk: usize,
    n: usize,
    ci: bool,
    out: &mut [Span],
) -> usize {
    let (mut w, mut open, mut skip) = (0usize, u32::MAX, 0usize);
    for bi in 0..nblk {
        let mut m = starts[bi];
        let f = flag[bi];
        if f == 0 && skip <= bi * 64 {
            while m != 0 {
                let pos = (bi * 64 + m.trailing_zeros() as usize) as u32;
                if open != u32::MAX {
                    out[w] = Span::new(open, pos);
                    w += 1;
                }
                open = pos;
                m &= m - 1;
            }
            continue;
        }
        while m != 0 {
            let j = m.trailing_zeros() as usize;
            let pos = bi * 64 + j;
            m &= m - 1;
            if pos < skip {
                continue;
            }
            if open != u32::MAX {
                out[w] = Span::new(open, pos as u32);
                w += 1;
            }
            open = pos as u32;
            if f >> j & 1 != 0 {
                // Contractions chain (`'re've`, `y'all'd've`): the char after one is a token start
                // in its own right, so keep matching until one fails rather than handing control
                // back to the bit algebra — whose start bit there we are about to skip.
                let mut p = pos;
                while p < n {
                    let l = contr_len(text, p, ci);
                    if l == 0 {
                        break;
                    }
                    out[w] = Span::new(p as u32, (p + l) as u32);
                    w += 1;
                    p += l;
                }
                if p > pos {
                    open = if p < n { p as u32 } else { u32::MAX };
                    // `+ 1`: the algebra usually also has a start bit exactly at `p` (the letter
                    // run resumes there) and we have just opened it — consuming it again would
                    // emit an empty span. `contr_len(p) == 0` here, so nothing is lost.
                    skip = p + 1;
                }
            }
        }
    }
    if open != u32::MAX {
        out[w] = Span::new(open, n as u32);
        w += 1;
    }
    w
}

/// Byte length of the contraction at `i` (2 or 3), or 0. `ci` picks cl100k/o200k's `(?i:)` form
/// over GPT-2's case-sensitive one.
#[inline]
fn contr_len(text: &[u8], i: usize, ci: bool) -> usize {
    let n = text.len();
    if i + 1 >= n || text[i] != b'\'' || text[i + 1] >= 0x80 {
        return 0;
    }
    let c1 = if ci { text[i + 1] | 0x20 } else { text[i + 1] };
    match c1 {
        b's' | b't' | b'm' | b'd' => 2,
        b'r' | b'v' | b'l' if i + 2 < n && text[i + 2] < 0x80 => {
            let c2 = if ci { text[i + 2] | 0x20 } else { text[i + 2] };
            usize::from((matches!(c1, b'r' | b'v') && c2 == b'e') || (c1 == b'l' && c2 == b'l')) * 3
        }
        _ => 0,
    }
}

/// Builder-only cost probe: runs just the byte→bitstream transpose over every block and folds the
/// streams so nothing is optimised away. The difference against a full grammar is what the
/// bitstream program itself costs.
#[doc(hidden)]
#[must_use]
pub fn build_only(text: &[u8], tags: &[u8]) -> u64 {
    let (mut acc, mut code, mut cjk) = (0u64, CODE_CONT, false);
    for base in (0..text.len()).step_by(64) {
        let len = (text.len() - base).min(64);
        let (b, c) = build_block::<true>(text, tags, base, len, &deepseek::LUT, code, cjk);
        code = c;
        cjk = b.cjk >> (len - 1) & 1 != 0;
        acc ^= b.cont ^ b.p0 ^ b.p1 ^ b.p2 ^ b.cjk;
    }
    acc
}

/// Convenience wrapper: classify + deepseek bitsplit over caller-owned scratch.
#[must_use]
pub fn pre_tokenize(text: &[u8], tags: &mut [u8], starts: &mut [u64], out: &mut [Span]) -> usize {
    classify::classify(text, tags);
    bitsplit_deepseek(text, tags, starts, out)
}
