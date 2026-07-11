//! Pure-Rust Unicode normalization — NFD, NFKD, NFC, NFKC — with no `unicode-normalization` at runtime.
//!
//! Unlike the tag-driven normalizers, this does NOT classify: it never materializes a per-byte tag
//! buffer. Decomposition (NFD/NFKD) bulk-*skips* runs that are already stable and only decomposes the
//! rest; composition (NFC/NFKC) uses a quick-check bitset to borrow already-normalized input untouched.
//! All predicates and tables are baked by `bitmap_gen` and byte-exact with `unicode-normalization`.
//!
//! Decompose hot path (aarch64): `skip_ascii` (`vmaxvq`, 16 B/iter) and `skip_clear_{2,3}byte`
//! (`vld2q`/`vld3q` deinterleave, 16 codepoints/iter) probe a 1-bit predicate in-lane and jump over
//! stable runs with no output. Cold path: Hangul by S_BASE arithmetic, everything else via the baked
//! index, accumulating ccc-marked chars and canonically reordering (stable sort on ccc) at each starter.
//!
//! The kernel is generic over zero-size `Form` markers whose associated consts ARE the concrete baked
//! `static` arrays. Monomorphization inlines those statics into every probe/lookup — a `&'static Tables`
//! struct passed at runtime instead measured ~40% slower on mark-dense scripts (the compiler can't see
//! which arrays are live, so it can't specialize the per-char decompose loop).
use std::borrow::Cow;
use std::cell::RefCell;

use crate::compose_tables::{COMPOSE, NFC_RELEVANT, NFC_RELEVANT_CAP, NFKC_RELEVANT, NFKC_RELEVANT_CAP};
use crate::nfd_tables::{NFD_CAP, NFD_DECOMP, NFD_TRIE_DATA, NFD_TRIE_INDEX, NFD_UNSTABLE};
use crate::nfkd_tables::{NFKD_CAP, NFKD_DECOMP, NFKD_TRIE_DATA, NFKD_TRIE_INDEX, NFKD_UNSTABLE};

thread_local! {
    /// Composition scratch: (decomposed (ccc,char) sequence, its reorder buffer, kept-marks buffer).
    static COMPOSE_SCRATCH: RefCell<(Vec<(u8, char)>, Vec<(u8, char)>, Vec<char>)> =
        const { RefCell::new((Vec::new(), Vec::new(), Vec::new())) };
}

const HANGUL: std::ops::RangeInclusive<u32> = 0xAC00..=0xD7A3;

// ── form markers ─────────────────────────────────────────────────────────────────────────────────

/// A predicate bitset over codepoints `< CAP` (`BITS.len() == CAP / 64`, `CAP` a multiple of 64, and
/// `CAP ≥ 0x10000`). Implemented by the decompose forms (their "unstable" bitset) and the NFC/NFKC
/// composition-relevance bitsets. Associated consts, so a generic user sees the concrete `static`.
trait Bitset {
    const BITS: &'static [u64];
    const CAP: u32;
}

/// A decomposition form (NFD canonical / NFKD compatibility): its "unstable" bitset plus the flattened
/// decomposition blob and the two-level lookup trie (`DATA[INDEX[cp>>6] + (cp&63)]`).
trait Decomp: Bitset {
    const DECOMP: &'static [u32];
    const TRIE_INDEX: &'static [u32];
    const TRIE_DATA: &'static [u32];

    /// Baked decomposition entries (`(ccc << 24) | cp` each) of `cp`, or `&[]` if absent — an O(1)
    /// two-level trie gather. Only ever called for unstable chars (bit set ⇒ `cp < CAP`), so all present.
    #[inline]
    fn entries(cp: u32) -> &'static [u32] {
        // SAFETY: cp < CAP (caller only decomposes unstable chars) ⇒ (cp>>6) < TRIE_INDEX.len(); the trie
        // is built so INDEX[blk] + (cp&63) is in range, and off + len ≤ DECOMP.len(), for every entry.
        unsafe {
            let slot = *Self::TRIE_INDEX.get_unchecked((cp >> 6) as usize) + (cp & 63);
            let packed = *Self::TRIE_DATA.get_unchecked(slot as usize);
            if packed == 0 {
                return &[];
            }
            let (off, len) = ((packed >> 8) as usize, (packed & 0xFF) as usize);
            Self::DECOMP.get_unchecked(off..off + len)
        }
    }
}

enum Nfd {}
enum Nfkd {}
enum NfcRelevant {}
enum NfkcRelevant {}

impl Bitset for Nfd {
    const BITS: &'static [u64] = &NFD_UNSTABLE;
    const CAP: u32 = NFD_CAP;
}
impl Decomp for Nfd {
    const DECOMP: &'static [u32] = &NFD_DECOMP;
    const TRIE_INDEX: &'static [u32] = &NFD_TRIE_INDEX;
    const TRIE_DATA: &'static [u32] = &NFD_TRIE_DATA;
}
impl Bitset for Nfkd {
    const BITS: &'static [u64] = &NFKD_UNSTABLE;
    const CAP: u32 = NFKD_CAP;
}
impl Decomp for Nfkd {
    const DECOMP: &'static [u32] = &NFKD_DECOMP;
    const TRIE_INDEX: &'static [u32] = &NFKD_TRIE_INDEX;
    const TRIE_DATA: &'static [u32] = &NFKD_TRIE_DATA;
}
impl Bitset for NfcRelevant {
    const BITS: &'static [u64] = &NFC_RELEVANT;
    const CAP: u32 = NFC_RELEVANT_CAP;
}
impl Bitset for NfkcRelevant {
    const BITS: &'static [u64] = &NFKC_RELEVANT;
    const CAP: u32 = NFKC_RELEVANT_CAP;
}

// ── bit probes ───────────────────────────────────────────────────────────────────────────────────

/// Is codepoint `cp` set in `B`'s bitset? Codepoints `≥ CAP` are never set. Hot: once per char.
#[inline]
fn bit_set<B: Bitset>(cp: u32) -> bool {
    // SAFETY: cp < CAP ⇒ (cp >> 6) < CAP / 64 == BITS.len() (CAP is a multiple of 64 by construction).
    cp < B::CAP && unsafe { (B::BITS.get_unchecked((cp >> 6) as usize) >> (cp & 63)) & 1 != 0 }
}

/// Same, for a BMP codepoint the SIMD lanes produce (`cp < 0x10000`; every bitset has `CAP ≥ 0x10000`).
#[inline]
fn bmp_bit<B: Bitset>(cp: u16) -> bool {
    // SAFETY: cp < 0x10000 ⇒ (cp >> 6) < 0x400 ≤ BITS.len() (CAP ≥ 0x10000 for every bitset).
    unsafe { (B::BITS.get_unchecked((cp >> 6) as usize) >> (cp & 63)) & 1 != 0 }
}

/// Decode the UTF-8 codepoint at `bytes[i]` → (codepoint, byte width). Faster than `str::chars().next()`
/// on the hot per-char path. `bytes` must be valid UTF-8 at a char boundary `i` (callers pass `str`).
#[inline]
fn decode_cp(bytes: &[u8], i: usize) -> (u32, usize) {
    let b0 = bytes[i];
    if b0 < 0x80 {
        return (b0 as u32, 1);
    }
    // SAFETY: valid UTF-8 ⇒ each multibyte lead is followed by its 1–3 continuation bytes in bounds.
    unsafe {
        let c = |k: usize| (*bytes.get_unchecked(i + k) & 0x3F) as u32;
        if b0 < 0xE0 {
            ((((b0 & 0x1F) as u32) << 6) | c(1), 2)
        } else if b0 < 0xF0 {
            ((((b0 & 0x0F) as u32) << 12) | (c(1) << 6) | c(2), 3)
        } else {
            ((((b0 & 0x07) as u32) << 18) | (c(1) << 12) | (c(2) << 6) | c(3), 4)
        }
    }
}

// ── SIMD stable-run skipping ─────────────────────────────────────────────────────────────────────

/// Advance over the maximal run of ASCII bytes from `i` — always stable/irrelevant. 16 B/iter on NEON.
#[inline]
fn skip_ascii(bytes: &[u8], mut i: usize) -> usize {
    let n = bytes.len();
    #[cfg(target_arch = "aarch64")]
    // SAFETY: every load is gated by `i + 16 <= n`; NEON loads are alignment-free.
    unsafe {
        use std::arch::aarch64::*;
        while i + 16 <= n {
            if vmaxvq_u8(vld1q_u8(bytes.as_ptr().add(i))) >= 0x80 {
                break;
            }
            i += 16;
        }
    }
    while i < n && bytes[i] < 0x80 {
        i += 1;
    }
    i
}

/// Advance over the leading run of **uniform 3-byte** codepoints (CJK & most non-Latin scripts) whose
/// bit is CLEAR in `B` — up to 16 codepoints/iter. `vld3q_u8` deinterleaves 48 bytes into lead/b1/b2
/// lanes (no shuffle table); the codepoint is computed in-lane and the bit gathered per lane. Stops at
/// the first lane that is NOT a clear 3-byte char (a set bit OR a different width, e.g. an ASCII space) —
/// advancing the uniform-3-byte PREFIX rather than bailing, so runs punctuated by spaces still progress.
#[cfg(target_arch = "aarch64")]
#[inline]
fn skip_clear_3byte<B: Bitset>(bytes: &[u8], mut i: usize) -> usize {
    use std::arch::aarch64::*;
    let n = bytes.len();
    // SAFETY: every load is gated by `i + 48 <= n`; NEON loads/stores are alignment-free.
    unsafe {
        while i + 48 <= n {
            let x = vld3q_u8(bytes.as_ptr().add(i)); // .0 = leads, .1 = b1, .2 = b2 (16 each)
            let lead_ok = vandq_u8(vcgeq_u8(x.0, vdupq_n_u8(0xE0)), vcleq_u8(x.0, vdupq_n_u8(0xEF)));
            let mut cps = [0u16; 16];
            let lo = (vget_low_u8(x.0), vget_low_u8(x.1), vget_low_u8(x.2));
            let hi = (vget_high_u8(x.0), vget_high_u8(x.1), vget_high_u8(x.2));
            for (h, (l8, b18, b28)) in [(0usize, lo), (8, hi)] {
                let l = vandq_u16(vmovl_u8(l8), vdupq_n_u16(0x0F));
                let b1 = vandq_u16(vmovl_u8(b18), vdupq_n_u16(0x3F));
                let b2 = vandq_u16(vmovl_u8(b28), vdupq_n_u16(0x3F));
                let cp = vorrq_u16(vorrq_u16(vshlq_n_u16::<12>(l), vshlq_n_u16::<6>(b1)), b2);
                vst1q_u16(cps.as_mut_ptr().add(h), cp);
            }
            let mut lok = [0u8; 16];
            vst1q_u8(lok.as_mut_ptr(), lead_ok);
            // stop at the first lane that isn't a clear 3-byte lead (short-circuits before the bit probe)
            match (0..16).position(|l| lok[l] != 0xFF || bmp_bit::<B>(cps[l])) {
                Some(lane) => return i + lane * 3,
                None => i += 48,
            }
        }
    }
    i
}

/// Same idea for the leading run of **uniform 2-byte** chars (Latin-1/Greek/Cyrillic/Arabic/…) via
/// `vld2q_u8`; stops at the first non-clear-2-byte lane so e.g. Cyrillic words punctuated by ASCII
/// spaces advance word-by-word instead of bailing on the whole 32-byte window.
#[cfg(target_arch = "aarch64")]
#[inline]
fn skip_clear_2byte<B: Bitset>(bytes: &[u8], mut i: usize) -> usize {
    use std::arch::aarch64::*;
    let n = bytes.len();
    // SAFETY: every load is gated by `i + 32 <= n`; NEON loads/stores are alignment-free.
    unsafe {
        while i + 32 <= n {
            let x = vld2q_u8(bytes.as_ptr().add(i)); // .0 = leads, .1 = conts (16 each)
            let lead_ok = vandq_u8(vcgeq_u8(x.0, vdupq_n_u8(0xC2)), vcleq_u8(x.0, vdupq_n_u8(0xDF)));
            let mut cps = [0u16; 16];
            let lo = (vget_low_u8(x.0), vget_low_u8(x.1));
            let hi = (vget_high_u8(x.0), vget_high_u8(x.1));
            for (h, (l8, c8)) in [(0usize, lo), (8, hi)] {
                let l = vandq_u16(vmovl_u8(l8), vdupq_n_u16(0x1F));
                let cc = vandq_u16(vmovl_u8(c8), vdupq_n_u16(0x3F));
                let cp = vorrq_u16(vshlq_n_u16::<6>(l), cc);
                vst1q_u16(cps.as_mut_ptr().add(h), cp);
            }
            let mut lok = [0u8; 16];
            vst1q_u8(lok.as_mut_ptr(), lead_ok);
            match (0..16).position(|l| lok[l] != 0xFF || bmp_bit::<B>(cps[l])) {
                Some(lane) => return i + lane * 2,
                None => i += 32,
            }
            i += 32;
        }
    }
    i
}

/// Find the next char at or after `i` whose bit is SET in `B`, returning `(index, codepoint, width)` —
/// or `(bytes.len(), 0, 0)` if none. Returning the DECODE lets the caller decompose it without decoding
/// twice. Checks the current char's bit FIRST (dense set-runs return immediately, no wasted SIMD probe);
/// a CLEAR non-ASCII char triggers a `vld2q`/`vld3q` bulk-skip of its uniform run; ASCII skips via `vmaxvq`.
#[inline]
fn next_set<B: Bitset>(bytes: &[u8], mut i: usize) -> (usize, u32, usize) {
    let n = bytes.len();
    while i < n {
        let b = bytes[i];
        if b < 0x80 {
            i = skip_ascii(bytes, i);
            continue;
        }
        let (cp, w) = decode_cp(bytes, i);
        if bit_set::<B>(cp) {
            return (i, cp, w);
        }
        // stable non-ASCII. Peek the next char scalar: if it's SET, the stable run is just this one char —
        // return the peeked char directly (reusing its decode) and skip the `vld2q`. This is the common
        // mark-dense shape (letter, mark, letter, mark, …) where a per-char SIMD probe is pure overhead.
        // Only a 2nd consecutive stable non-ASCII char escalates to the SIMD bulk-skip (long runs: CJK/…).
        #[cfg(target_arch = "aarch64")]
        if i + w < n && bytes[i + w] >= 0x80 {
            let (cp2, w2) = decode_cp(bytes, i + w);
            if bit_set::<B>(cp2) {
                return (i + w, cp2, w2);
            }
            let ni = match w {
                2 => skip_clear_2byte::<B>(bytes, i),
                3 => skip_clear_3byte::<B>(bytes, i),
                _ => i,
            };
            if ni > i {
                i = ni;
                continue;
            }
        }
        i += w;
    }
    (n, 0, 0)
}

// ── decomposition (NFD / NFKD) ───────────────────────────────────────────────────────────────────

/// Decompose output is written DIRECTLY to `out` (no deferred reorder buffer): a char's own NFD is
/// already canonically ordered, and cross-char reordering is rare, so the common path is a plain push +
/// a `last_ccc` compare. `run_start` is the byte offset in `out` where the current combining sequence's
/// marks began; `last_ccc` is the ccc of the last mark written (0 after a starter).
struct Emit {
    last_ccc: u8,
    run_start: usize,
}

/// A ccc-0 starter (base char / Hangul jamo / ASCII): closes the previous combining sequence.
#[inline]
fn push_starter(out: &mut String, ch: char, e: &mut Emit) {
    out.push(ch);
    e.last_ccc = 0;
    e.run_start = out.len();
}

/// A combining mark (ccc `cc` ≠ 0): in canonical order (`cc ≥ last_ccc`) it appends; otherwise the cold
/// path inserts it at the right spot in the current mark run (rare in real text).
#[inline]
fn push_mark<D: Decomp>(out: &mut String, cc: u8, ch: char, e: &mut Emit) {
    if cc >= e.last_ccc {
        out.push(ch);
        e.last_ccc = cc;
    } else {
        reorder_insert::<D>(out, e.run_start, cc, ch); // last_ccc stays the run's max
    }
}

/// Insert combining mark `ch` (ccc `cc`) into the already-written, canonically-ordered mark run at
/// `out[run_start..]`, before the first written mark whose ccc exceeds `cc` (ccc via the trie). Only hit
/// when a mark arrives out of canonical order — cold.
#[cold]
fn reorder_insert<D: Decomp>(out: &mut String, run_start: usize, cc: u8, ch: char) {
    let mut pos = out.len();
    let mut byte = run_start;
    for c in out[run_start..].chars() {
        let e = D::entries(c as u32);
        let c_ccc = if e.is_empty() { 0 } else { (e[0] >> 24) as u8 };
        if c_ccc > cc {
            pos = byte;
            break;
        }
        byte += c.len_utf8();
    }
    out.insert(pos, ch);
}

/// Write the decomposition of unstable char `cp` directly to `out`. Hangul is arithmetic; else the trie.
#[inline]
fn decompose_char<D: Decomp>(cp: u32, out: &mut String, e: &mut Emit) {
    if HANGUL.contains(&cp) {
        let s = cp - 0xAC00;
        // SAFETY: the three jamo blocks are valid scalar values; jamo are ccc-0 starters.
        unsafe {
            push_starter(out, char::from_u32_unchecked(0x1100 + s / 588), e);
            push_starter(out, char::from_u32_unchecked(0x1161 + (s % 588) / 28), e);
            let tt = s % 28;
            if tt != 0 {
                push_starter(out, char::from_u32_unchecked(0x11A7 + tt), e);
            }
        }
    } else {
        for &en in D::entries(cp) {
            let cc = (en >> 24) as u8;
            // SAFETY: baked from valid chars (bitmap_gen round-trips every entry).
            let ch = unsafe { char::from_u32_unchecked(en & 0xFF_FFFF) };
            if cc == 0 {
                push_starter(out, ch, e);
            } else {
                push_mark::<D>(out, cc, ch, e);
            }
        }
    }
}

/// Is `input` ALREADY in form `D` (`str::nfd()` would be the identity)? Then the caller borrows it —
/// crucial because most real text is already normalized, including scripts my "unstable" bitset flags
/// wholesale: e.g. Arabic/Hebrew whose combining marks all carry ccc ≠ 0 but are in canonical order. A
/// char breaks normalization iff it DECOMPOSES (trie entry ≠ `[itself]`, incl. Hangul's empty entry) or
/// it's a mark whose ccc drops below the preceding mark's (out of canonical order). Same SIMD skipping as
/// the owned path, so already-NFD text is one fast pass with no allocation.
fn is_already_normalized<D: Decomp>(bytes: &[u8]) -> bool {
    let n = bytes.len();
    let mut i = 0;
    let mut last_ccc = 0u8; // ccc of the previous mark in the current combining sequence (0 after a starter)
    while i < n {
        let b = bytes[i];
        if b < 0x80 {
            last_ccc = 0;
            i = skip_ascii(bytes, i);
            continue;
        }
        let (cp, w) = decode_cp(bytes, i);
        if !bit_set::<D>(cp) {
            last_ccc = 0; // a ccc-0 starter closes the combining sequence
            #[cfg(target_arch = "aarch64")]
            if i + w < n && bytes[i + w] >= 0x80 {
                let (cp2, _) = decode_cp(bytes, i + w);
                if !bit_set::<D>(cp2) {
                    let ni = match w {
                        2 => skip_clear_2byte::<D>(bytes, i),
                        3 => skip_clear_3byte::<D>(bytes, i),
                        _ => i,
                    };
                    if ni > i {
                        i = ni;
                        continue;
                    }
                }
            }
            i += w;
            continue;
        }
        // flagged char: decomposes, or a mark that might be out of order
        let e = D::entries(cp);
        if e.len() != 1 || (e[0] & 0xFF_FFFF) != cp {
            return false; // decomposes (content change; Hangul's empty entry lands here too)
        }
        let ccc = (e[0] >> 24) as u8;
        if ccc < last_ccc {
            return false; // reorderable mark out of canonical order
        }
        last_ccc = ccc;
        i += w;
    }
    true
}

/// Decompose `input` under form `D`. `Cow::Borrowed` when already in that form (the common case — see
/// [`is_already_normalized`]), else owned. The owned path decomposes each unstable char DIRECTLY to `out`
/// (reusing `next_set`'s decode — one decode per char) and lets `next_set` bulk-skip+copy the stable runs
/// between them (`vmaxvq`/`vld2q`/`vld3q`). No deferred reorder buffer: a stable run begins with a ccc-0
/// starter, so it just resets `Emit`. SIMD-decoding whole chunks and a `u64` memcpy-emit were both tried
/// and regressed (dispatch / cache) — see the module note.
fn decompose<'a, D: Decomp>(input: &'a str) -> Cow<'a, str> {
    let bytes = input.as_bytes();
    let n = bytes.len();
    if is_already_normalized::<D>(bytes) {
        return Cow::Borrowed(input);
    }
    let (first, mut cp, mut w) = next_set::<D>(bytes, 0);
    let mut out = String::with_capacity(n + (n >> 4));
    out.push_str(&input[..first]);
    let mut e = Emit { last_ccc: 0, run_start: out.len() };
    let mut i = first;
    loop {
        // (cp, w) is the already-decoded unstable char at i (from next_set) — never re-decoded
        decompose_char::<D>(cp, &mut out, &mut e);
        i += w;
        let (ns, ncp, nw) = next_set::<D>(bytes, i);
        if ns > i {
            // a stable run followed: it starts with a ccc-0 starter, so the combining sequence is closed
            out.push_str(&input[i..ns]);
            e.last_ccc = 0;
            e.run_start = out.len();
        }
        if ns == n {
            break;
        }
        (i, cp, w) = (ns, ncp, nw);
    }
    Cow::Owned(out)
}

/// NFD-normalize `input`. Byte-exact with `str::nfd()`.
pub fn nfd(input: &str) -> Cow<'_, str> {
    decompose::<Nfd>(input)
}

/// NFKD-normalize `input`. Byte-exact with `str::nfkd()`.
pub fn nfkd(input: &str) -> Cow<'_, str> {
    decompose::<Nfkd>(input)
}

/// NFD-decompose a single `char`, invoking `f` with each output char in canonical order (a stable char
/// calls `f(c)` once). A single char's decomposition is already canonically ordered, so callers that
/// decompose char-by-char (e.g. BertNormalizer's `strip_accents`) need no cross-char reordering. Uses the
/// baked NFD trie + arithmetic Hangul — no `unicode-normalization` at runtime. Byte-exact with `char::nfd()`.
#[inline]
pub fn nfd_char(c: char, mut f: impl FnMut(char)) {
    let cp = c as u32;
    if HANGUL.contains(&cp) {
        let s = cp - 0xAC00;
        // SAFETY: the three jamo blocks are valid scalar values by construction.
        unsafe {
            f(char::from_u32_unchecked(0x1100 + s / 588));
            f(char::from_u32_unchecked(0x1161 + (s % 588) / 28));
            let t = s % 28;
            if t != 0 {
                f(char::from_u32_unchecked(0x11A7 + t));
            }
        }
    } else if bit_set::<Nfd>(cp) {
        for &e in Nfd::entries(cp) {
            // SAFETY: baked from valid chars (bitmap_gen round-trips every entry).
            f(unsafe { char::from_u32_unchecked(e & 0xFF_FFFF) });
        }
    } else {
        f(c); // stable → itself
    }
}

// ── composition (NFC / NFKC) ─────────────────────────────────────────────────────────────────────

/// Canonical primary composite of a starter `a` and following char `b`, if any. Hangul L+V / LV+T are
/// arithmetic; everything else is the baked `COMPOSE` table (canonical composites minus exclusions).
#[inline]
fn primary_composite(a: char, b: char) -> Option<char> {
    let (ai, bi) = (a as u32, b as u32);
    if (0x1100..=0x1112).contains(&ai) && (0x1161..=0x1175).contains(&bi) {
        // SAFETY: result is a valid Hangul syllable by construction.
        return Some(unsafe { char::from_u32_unchecked(0xAC00 + ((ai - 0x1100) * 21 + (bi - 0x1161)) * 28) });
    }
    if HANGUL.contains(&ai) && (ai - 0xAC00) % 28 == 0 && (0x11A8..=0x11C2).contains(&bi) {
        // SAFETY: LV + T stays within the Hangul syllable block.
        return Some(unsafe { char::from_u32_unchecked(ai + (bi - 0x11A7)) });
    }
    let key = ((ai as u64) << 21) | bi as u64;
    COMPOSE
        .binary_search_by_key(&key, |&(k, _)| k)
        .ok()
        // SAFETY: baked composites are valid scalar values.
        .map(|p| unsafe { char::from_u32_unchecked(COMPOSE[p].1) })
}

#[inline]
fn emit_pair(cc: u8, ch: char, pending: &mut Vec<(u8, char)>, buf: &mut Vec<(u8, char)>) {
    if cc == 0 {
        flush_pairs(pending, buf);
        buf.push((0, ch));
    } else {
        pending.push((cc, ch));
    }
}

#[inline]
fn flush_pairs(pending: &mut Vec<(u8, char)>, buf: &mut Vec<(u8, char)>) {
    if pending.len() > 1 {
        pending.sort_by_key(|&(cc, _)| cc);
    }
    buf.extend(pending.drain(..));
}

/// Fully decompose `input` under form `D` into `buf` as `(ccc, char)` pairs in canonical order (the
/// NFD/NFKD form). No skipping — composition needs the whole sequence, including stable starters.
fn decompose_to_pairs<D: Decomp>(
    input: &str,
    buf: &mut Vec<(u8, char)>,
    pending: &mut Vec<(u8, char)>,
) {
    buf.clear();
    pending.clear();
    for c in input.chars() {
        let cp = c as u32;
        if !bit_set::<D>(cp) {
            emit_pair(0, c, pending, buf);
        } else if HANGUL.contains(&cp) {
            let s = cp - 0xAC00;
            // SAFETY: valid jamo by construction; jamo are ccc-0 starters.
            unsafe {
                emit_pair(0, char::from_u32_unchecked(0x1100 + s / 588), pending, buf);
                emit_pair(0, char::from_u32_unchecked(0x1161 + (s % 588) / 28), pending, buf);
                let tt = s % 28;
                if tt != 0 {
                    emit_pair(0, char::from_u32_unchecked(0x11A7 + tt), pending, buf);
                }
            }
        } else {
            for &e in D::entries(cp) {
                // SAFETY: baked valid char.
                emit_pair((e >> 24) as u8, unsafe { char::from_u32_unchecked(e & 0xFF_FFFF) }, pending, buf);
            }
        }
    }
    flush_pairs(pending, buf);
}

/// Canonically compose the decomposed sequence `seq` (`(ccc, char)`) into `out` (UAX #15 composition:
/// combine each starter with following non-blocked composables). `marks` is a reused scratch buffer.
fn compose_into(seq: &[(u8, char)], out: &mut String, marks: &mut Vec<char>) {
    let n = seq.len();
    let mut i = 0;
    while i < n {
        let (cc0, first) = seq[i];
        if cc0 != 0 {
            out.push(first); // stray leading combining mark — nothing to compose onto
            i += 1;
            continue;
        }
        let mut cur = first; // the starter, possibly replaced by composites
        let mut last_ccc: i16 = -1; // ccc of the last KEPT char since the starter; -1 = none kept
        marks.clear();
        let mut j = i + 1;
        while j < n {
            let (cc, c) = seq[j];
            let cc = cc as i16;
            // not blocked: no kept char since the starter has ccc == 0 (impossible mid-run) or ccc >= cc
            let not_blocked = if cc == 0 { last_ccc == -1 } else { last_ccc < cc };
            if not_blocked {
                if let Some(comp) = primary_composite(cur, c) {
                    cur = comp; // c consumed; last_ccc unchanged (kept marks are the only blockers)
                    j += 1;
                    continue;
                }
            }
            if cc == 0 {
                break; // an uncomposable starter begins the next run
            }
            last_ccc = cc;
            marks.push(c);
            j += 1;
        }
        out.push(cur);
        for &m in marks.iter() {
            out.push(m);
        }
        i = j;
    }
}

/// Compose `input` (borrow if already normalized) using decomposition form `D` and relevance bitset `R`.
/// Shared by NFC (`Nfd` + `NfcRelevant`) and NFKC (`Nfkd` + `NfkcRelevant`).
fn compose<'a, D: Decomp, R: Bitset>(input: &'a str) -> Cow<'a, str> {
    let bytes = input.as_bytes();
    let n = bytes.len();
    // No composition-relevant char (no ccc != 0, no QC != Yes) ⇒ already in the target form.
    if next_set::<R>(bytes, 0).0 == n {
        return Cow::Borrowed(input);
    }
    // ponytail: recompose the whole string (matches unicode-normalization's non-quick path). A windowed
    // recompose from the last starter before the first relevant char would cut work on long mostly-NFC
    // docs — add if a profile shows it matters.
    let mut out = String::with_capacity(n);
    COMPOSE_SCRATCH.with(|sc| {
        let (buf, pending, marks) = &mut *sc.borrow_mut();
        decompose_to_pairs::<D>(input, buf, pending);
        compose_into(buf, &mut out, marks);
    });
    Cow::Owned(out)
}

/// NFC-normalize `input`. Byte-exact with `str::nfc()`.
pub fn nfc(input: &str) -> Cow<'_, str> {
    compose::<Nfd, NfcRelevant>(input)
}

/// NFKC-normalize `input`. Byte-exact with `str::nfkc()`.
pub fn nfkc(input: &str) -> Cow<'_, str> {
    compose::<Nfkd, NfkcRelevant>(input)
}

#[cfg(test)]
mod tests {
    use super::{nfc, nfd, nfkc, nfkd};
    use unicode_normalization::UnicodeNormalization;

    const CORPUS: &[&str] = &[
        "hello world 12345",
        "café crème déjà",             // Latin accents
        "Ἀρχαία ἑλληνικά",             // polytonic Greek
        "한국어 테스트 문장입니다",     // Hangul
        "\u{1100}\u{1161}\u{11A8}",    // isolated jamo L+V+T → should compose to a syllable (NFC)
        "Å Å ẛ̣ ǰ",                    // multi-step + reorder
        "e\u{0301}\u{0323}",           // reorder (ccc 230 vs 220) + compose
        "a\u{0323}\u{0301}",           // canonical-order swap
        "\u{fb01} ² ½ ﷺ",             // compatibility (NFKD/NFKC only)
        "ﬁ ² Ⅻ ㍿",                   // more compat
        "mixed 世界 café Москва テスト",
        "\u{2F800}",                   // astral CJK-compat ideograph
        "",
    ];

    #[test]
    fn nfd_matches() {
        for s in CORPUS {
            assert_eq!(nfd(s), s.nfd().collect::<String>(), "NFD {s:?}");
        }
    }
    #[test]
    fn nfkd_matches() {
        for s in CORPUS {
            assert_eq!(nfkd(s), s.nfkd().collect::<String>(), "NFKD {s:?}");
        }
    }
    #[test]
    fn nfc_matches() {
        for s in CORPUS {
            assert_eq!(nfc(s), s.nfc().collect::<String>(), "NFC {s:?}");
        }
    }
    #[test]
    fn nfkc_matches() {
        for s in CORPUS {
            assert_eq!(nfkc(s), s.nfkc().collect::<String>(), "NFKC {s:?}");
        }
    }

    #[test]
    fn borrows_when_normalized() {
        use std::borrow::Cow;
        assert!(matches!(nfd("这是中文"), Cow::Borrowed(_)));
        assert!(matches!(nfc("already NFC ascii"), Cow::Borrowed(_)));
        assert!(matches!(nfc("café"), Cow::Borrowed(_))); // precomposed é is NFC-stable
    }

    /// Every codepoint the tables cover, isolated and glued to combining marks, for all four forms —
    /// byte-exact with `unicode-normalization`.
    #[test]
    fn exhaustive() {
        let mut buf = String::new();
        for cp in 0u32..0x30000 {
            let Some(c) = char::from_u32(cp) else { continue };
            for suffix in ["", "\u{0301}", "\u{0323}\u{0301}"] {
                buf.clear();
                buf.push(c);
                buf.push_str(suffix);
                assert_eq!(nfd(&buf), buf.nfd().collect::<String>(), "NFD {cp:#x}");
                assert_eq!(nfkd(&buf), buf.nfkd().collect::<String>(), "NFKD {cp:#x}");
                assert_eq!(nfc(&buf), buf.nfc().collect::<String>(), "NFC {cp:#x}");
                assert_eq!(nfkc(&buf), buf.nfkc().collect::<String>(), "NFKC {cp:#x}");
            }
        }
    }
}
