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
use crate::nfd_tables::{NFD_CAP, NFD_DECOMP, NFD_MAX_EXPAND, NFD_TRIE_DATA, NFD_TRIE_INDEX, NFD_UNSTABLE};
use crate::nfkd_tables::{
    NFKD_CAP, NFKD_DECOMP, NFKD_MAX_EXPAND, NFKD_TRIE_DATA, NFKD_TRIE_INDEX, NFKD_UNSTABLE,
};

thread_local! {
    /// Composition scratch: (decomposed (ccc,char) sequence, its reorder buffer, kept-marks buffer).
    static COMPOSE_SCRATCH: RefCell<(Vec<(u8, char)>, Vec<(u8, char)>, Vec<char>)> =
        const { RefCell::new((Vec::new(), Vec::new(), Vec::new())) };
    /// Norm-classify tag scratch for the whole-buffer decompose (aarch64): one tag byte per input byte,
    /// reused across calls so steady-state normalization does no allocation.
    static TAG_SCRATCH: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
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
    /// Byte-form twin of the trie (generated — see `gen_byte_tables`): `BYTE_DATA` is slot-parallel to
    /// `TRIE_DATA`; a non-zero slot is `(off << 8) | byte_len` with `BYTE_BLOB[off..off+len]` the
    /// decomposition's UTF-8 bytes and `BYTE_BLOB[off-3..off]` = `[first_ccc, last_ccc, mark_run_off]`.
    /// The SIMD decompose kernel copies these blobs directly instead of re-encoding `(ccc, char)` entries.
    const BYTE_DATA: &'static [u32];
    const BYTE_BLOB: &'static [u8];
    /// `output.len() ≤ input.len() * MAX_EXPAND` — lets the owned path reserve once, realloc-free.
    const MAX_EXPAND: usize;
    /// Does check-tag `t` mean "this char CHANGES under this form"? (see [`check_tag`]): NFD breaks
    /// only on `0x7E`; NFKD additionally on the compat values `0x3C`/`0x3D`.
    fn tag_breaks(t: u8) -> bool;

    /// Byte-form entry of `cp`: `(blob_off, byte_len)` — or `(0, 0)` if absent (stable, or Hangul whose
    /// decomposition is arithmetic). `byte_len == 0xFF` is the scalar-fallback sentinel.
    #[inline]
    fn byte_entry(cp: u32) -> (usize, usize) {
        // SAFETY: same trie shape as `entries` — cp < CAP (callers only probe unstable chars).
        unsafe {
            let slot = *Self::TRIE_INDEX.get_unchecked((cp >> 6) as usize) + (cp & 63);
            let packed = *Self::BYTE_DATA.get_unchecked(slot as usize);
            ((packed >> 8) as usize, (packed & 0xFF) as usize)
        }
    }

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
    const BYTE_DATA: &'static [u32] = &crate::nfd_byte_tables::NFD_BYTE_DATA;
    const BYTE_BLOB: &'static [u8] = &crate::nfd_byte_tables::NFD_BYTE_BLOB;
    const MAX_EXPAND: usize = NFD_MAX_EXPAND;
    #[inline(always)]
    fn tag_breaks(t: u8) -> bool {
        t >= 0x7D // 0x7D/0x7E: canonical decomposition changes it
    }
}
impl Bitset for Nfkd {
    const BITS: &'static [u64] = &NFKD_UNSTABLE;
    const CAP: u32 = NFKD_CAP;
}
impl Decomp for Nfkd {
    const DECOMP: &'static [u32] = &NFKD_DECOMP;
    const TRIE_INDEX: &'static [u32] = &NFKD_TRIE_INDEX;
    const TRIE_DATA: &'static [u32] = &NFKD_TRIE_DATA;
    const BYTE_DATA: &'static [u32] = &crate::nfd_byte_tables::NFKD_BYTE_DATA;
    const BYTE_BLOB: &'static [u8] = &crate::nfd_byte_tables::NFKD_BYTE_BLOB;
    const MAX_EXPAND: usize = NFKD_MAX_EXPAND;
    #[inline(always)]
    fn tag_breaks(t: u8) -> bool {
        t >= 0x7D || (t & !1) == 0x3C // canonical (0x7D/0x7E) or compat (0x3C/0x3D) change
    }
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
fn skip_clear_3byte<B: Bitset>(bytes: &[u8], mut i: usize) -> (usize, u32) {
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
                Some(lane) => {
                    // stopped by a SET bit on a valid 3-byte lead: hand the already-decoded codepoint
                    // to the caller so it doesn't decode the same char again (one decode per cycle
                    // matters on set-dense CJK text like literary Japanese).
                    let cp = if lok[lane] == 0xFF { cps[lane] as u32 } else { 0 };
                    return (i + lane * 3, cp);
                }
                None => i += 48,
            }
        }
    }
    (i, 0)
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
        }
    }
    i
}

/// Advance over the maximal run of composed Hangul syllables (`U+AC00..=U+D7A3`) AND ASCII from `i`
/// — BYTE-class compares on 16-byte chunks, so the space between Korean words rides along instead of
/// bouncing back to the caller per word. A byte passes iff it is ASCII, a continuation (`80..BF`), or
/// a syllable lead (`EB|EC` free; `EA` requires next ≥ `B0`; `ED` requires next ≤ `9D` — the same
/// ranges the classify kernel uses; the rare `U+D780..=U+D7A3` tail falls out to the caller). Valid
/// UTF-8 input makes the class test sound: continuations only ever follow the leads we accepted.
/// Exits on a char boundary (backs up over continuations at the first failing byte).
#[cfg(target_arch = "aarch64")]
#[inline]
fn skip_hangul_or_ascii(bytes: &[u8], mut i: usize) -> usize {
    use std::arch::aarch64::*;
    let n = bytes.len();
    let start = i;
    // SAFETY: loads gated by `i + 16 <= n`; NEON loads are alignment-free.
    unsafe {
        let mut carry: u8 = 0; // last byte of the previous chunk (0 ⇒ unconstrained lane 0)
        while i + 16 <= n {
            let v = vld1q_u8(bytes.as_ptr().add(i));
            let ascii = vcltq_u8(v, vdupq_n_u8(0x80));
            let cont = vandq_u8(vcgeq_u8(v, vdupq_n_u8(0x80)), vcltq_u8(v, vdupq_n_u8(0xC0)));
            let lead = vandq_u8(vcgeq_u8(v, vdupq_n_u8(0xEA)), vcleq_u8(v, vdupq_n_u8(0xED)));
            let class_ok = vorrq_u8(vorrq_u8(ascii, cont), lead);
            // pair constraint: prev == EA ⇒ cur ≥ B0 ; prev == ED ⇒ cur ≤ 9D
            let prev = vextq_u8::<15>(vdupq_n_u8(carry), v);
            let bad_ea = vandq_u8(vceqq_u8(prev, vdupq_n_u8(0xEA)), vcltq_u8(v, vdupq_n_u8(0xB0)));
            let bad_ed = vandq_u8(vceqq_u8(prev, vdupq_n_u8(0xED)), vcgtq_u8(v, vdupq_n_u8(0x9D)));
            let ok = vbicq_u8(class_ok, vorrq_u8(bad_ea, bad_ed));
            if vminvq_u8(ok) == 0xFF {
                carry = vgetq_lane_u8::<15>(v);
                i += 16;
                continue;
            }
            let mut m = [0u8; 16];
            vst1q_u8(m.as_mut_ptr(), ok);
            let mut bad = i;
            for (l, &x) in m.iter().enumerate() {
                if x != 0xFF {
                    bad = i + l;
                    break;
                }
            }
            // back up to the char boundary (a failing continuation invalidates its whole char)
            while bad > start && bytes[bad] & 0xC0 == 0x80 {
                bad -= 1;
            }
            return bad;
        }
    }
    // scalar tail: whole syllables or ASCII only
    loop {
        if i >= n {
            return i;
        }
        let b = bytes[i];
        if b < 0x80 {
            i += 1;
            continue;
        }
        if i + 3 <= n && (0xEA..=0xED).contains(&b) {
            let cp = (((b & 0x0F) as u32) << 12)
                | (((bytes[i + 1] & 0x3F) as u32) << 6)
                | ((bytes[i + 2] & 0x3F) as u32);
            if HANGUL.contains(&cp) {
                i += 3;
                continue;
            }
        }
        return i;
    }
}

/// Write-through twin of [`skip_clear_3byte`] for the OWNED path: verified stable lanes are stored
/// back (one `vst3q` re-interleaves the registers the check already loaded — over-store rides the +48
/// capacity slack), so stable CJK/kana spans between emits need no separate copy pass at all.
/// Returns `(new_i, cp)`; `cp != 0` hands over the decoded SET char the scan stopped at.
#[cfg(target_arch = "aarch64")]
#[inline]
fn skip_clear_3byte_copy<B: Bitset>(bytes: &[u8], mut i: usize, out: &mut String) -> (usize, u32) {
    use std::arch::aarch64::*;
    let n = bytes.len();
    // SAFETY: loads gated by `i + 48 <= n`; stores ride reserved capacity (`+48` slack); only verified
    // whole-char bytes advance `len`, so the String stays valid UTF-8.
    unsafe {
        let v = out.as_mut_vec();
        let mut len = v.len();
        while i + 48 <= n {
            let x = vld3q_u8(bytes.as_ptr().add(i));
            vst3q_u8(v.as_mut_ptr().add(len), x); // unconditional; advance only by what verifies
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
            match (0..16).position(|l| lok[l] != 0xFF || bmp_bit::<B>(cps[l])) {
                Some(lane) => {
                    len += lane * 3;
                    v.set_len(len);
                    let cp = if lok[lane] == 0xFF { cps[lane] as u32 } else { 0 };
                    return (i + lane * 3, cp);
                }
                None => {
                    len += 48;
                    i += 48;
                }
            }
        }
        v.set_len(len);
    }
    (i, 0)
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
        // Composed-Hangul run (bit clear for the composition-relevance bitsets): range-skip whole
        // syllable runs with an in-lane compare — no per-cp bitset gather. Korean NFC/NFKC text is
        // almost entirely such runs.
        #[cfg(target_arch = "aarch64")]
        if HANGUL.contains(&cp) {
            let ni = skip_hangul_or_ascii(bytes, i);
            if ni > i {
                i = ni;
                continue;
            }
        }
        // stable non-ASCII. For 2-byte chars, peek the next char scalar: if it's SET, the stable run is
        // just this one char — return the peeked char directly and skip the `vld2q` (the common
        // mark-dense shape: letter, mark, letter, mark, …). 3-byte chars go STRAIGHT to the SIMD skip:
        // two scalar peek-decodes cost as much as the `vld3q` chunk, and CJK/kana gaps are chunk-sized.
        #[cfg(target_arch = "aarch64")]
        {
            if w == 3 {
                let (ni, ncp) = skip_clear_3byte::<B>(bytes, i);
                if ncp != 0 {
                    return (ni, ncp, 3); // the skip already decoded the SET char it stopped at
                }
                if ni > i {
                    i = ni;
                    continue;
                }
            } else if w == 2 && i + w < n && bytes[i + w] >= 0x80 {
                let (cp2, w2) = decode_cp(bytes, i + w);
                if bit_set::<B>(cp2) {
                    return (i + w, cp2, w2);
                }
                let ni = skip_clear_2byte::<B>(bytes, i);
                if ni > i {
                    i = ni;
                    continue;
                }
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

/// Byte index of the first char that BREAKS form `D` — or `bytes.len()` if the input is already fully
/// normalized (then the caller borrows it — crucial because most real text is already normalized,
/// including scripts my "unstable" bitset flags wholesale: e.g. Arabic/Hebrew whose combining marks all
/// carry ccc ≠ 0 but are in canonical order). A char breaks normalization iff it DECOMPOSES (trie entry
/// ≠ `[itself]`, incl. Hangul's empty entry) or it's a mark whose ccc drops below the preceding mark's
/// (out of canonical order). Same SIMD skipping as the owned path — and returning the *position* (not a
/// bool) lets the owned path copy the verified prefix wholesale instead of re-scanning it.
fn normalized_prefix<D: Decomp>(bytes: &[u8]) -> usize {
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
                        3 => skip_clear_3byte::<D>(bytes, i).0,
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
            return i; // decomposes (content change; Hangul's empty entry lands here too)
        }
        let ccc = (e[0] >> 24) as u8;
        if ccc < last_ccc {
            return i; // reorderable mark out of canonical order
        }
        last_ccc = ccc;
        i += w;
    }
    n
}

// ── whole-buffer SIMD decompose (aarch64) ────────────────────────────────────────────────────────
//
// The owned path's kernel: instead of `next_set`-skipping stable runs and decomposing unstable chars
// one at a time through `String::push`, process the buffer in fixed uniform-width chunks — 16 codepoints
// per iteration via `vld2q`/`vld3q` deinterleaved decode — and EMIT in bulk: runs of clear (stable) lanes
// are one memcpy, each set lane is one unaligned 16-byte blob copy (the generated byte-form tables), and
// Hangul is arithmetic with direct jamo byte stores. Short runs cost nothing extra: the chunk loop is
// entered per 16-codepoint window, not per run, which is what the per-run scalar batching attempts missed.
// Canonical-order across chars is guarded by the blob headers' `[first_ccc, last_ccc, mark_run_off]`
// chain check; anything irregular (out-of-order mark, > 16-byte decomposition, astral, chunk tails) takes
// the scalar `decompose_char` path, so the fast path never has to be clever — just byte-exact.

/// Append exactly `len` bytes from `src` with NO `memcpy` call and NO over-read: 16-byte moves plus an
/// OVERLAPPING tail block (the classic small-copy — every read stays inside `[src, src+len)`). Real
/// stable spans are words (4–50 bytes), where the call overhead of `push_str`'s memcpy dominates.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn copy_small(out: &mut String, src: *const u8, len: usize) {
    // SAFETY (caller): out.capacity() ≥ out.len() + len; src has exactly `len` readable bytes; the bytes
    // are valid UTF-8 at char boundaries (verbatim input spans).
    unsafe {
        let v = out.as_mut_vec();
        let ol = v.len();
        debug_assert!(ol + len <= v.capacity());
        let dst = v.as_mut_ptr().add(ol);
        if len >= 16 {
            let mut k = 0;
            while k + 16 <= len {
                std::ptr::copy_nonoverlapping(src.add(k), dst.add(k), 16);
                k += 16;
            }
            if k < len {
                // overlapping 16-byte tail: reads [src+len-16, src+len) — in bounds since len ≥ 16
                std::ptr::copy_nonoverlapping(src.add(len - 16), dst.add(len - 16), 16);
            }
        } else if len >= 8 {
            std::ptr::copy_nonoverlapping(src, dst, 8);
            std::ptr::copy_nonoverlapping(src.add(len - 8), dst.add(len - 8), 8);
        } else if len >= 4 {
            std::ptr::copy_nonoverlapping(src, dst, 4);
            std::ptr::copy_nonoverlapping(src.add(len - 4), dst.add(len - 4), 4);
        } else {
            for k in 0..len {
                *dst.add(k) = *src.add(k);
            }
        }
        v.set_len(ol + len);
    }
}

/// Append `copy_len` bytes from `src` (may over-copy up to 16 — capacity reserves `+16` for that) but
/// advance the length by exactly `adv`. The write primitive of the fast emit: blob/Hangul emits pass
/// `copy_len = 16, adv = real_len` so the store is one unaligned 16-byte move, never a `memcpy` call.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn raw_extend(out: &mut String, src: *const u8, copy_len: usize, adv: usize) {
    // SAFETY (caller): out.capacity() ≥ out.len() + max(copy_len, adv); src has copy_len readable bytes;
    // the adv bytes written are valid UTF-8 (blob entries and jamo always are).
    unsafe {
        let v = out.as_mut_vec();
        let len = v.len();
        debug_assert!(len + copy_len.max(adv) <= v.capacity());
        std::ptr::copy_nonoverlapping(src, v.as_mut_ptr().add(len), copy_len);
        v.set_len(len + adv);
    }
}

/// Emit one run of consecutive SET (unstable) chars starting at `i` (`cp`/`w` already decoded by the
/// caller), returning the index past the run. The hot emit is branch-light and push-free:
///   * Hangul (set bit + empty trie entry) — arithmetic L·V·(T), jamo bytes stored directly;
///   * everything else — ONE unaligned 16-byte copy from the generated byte-form blob (`BYTE_BLOB`),
///     guarded by the `[first_ccc, last_ccc, mark_run_off]` header chain check;
///   * out-of-canonical-order marks and > 16-byte decompositions (rare) — scalar [`decompose_char`].
/// Dense set runs (Hangul text, Greek accents, Hebrew/Arabic mark clusters) stay in this loop without
/// re-entering `next_set`'s scan machinery; the stable text between runs is memcpy'd once by the caller.
#[cfg(target_arch = "aarch64")]
#[inline]
fn emit_set<D: Decomp>(
    bytes: &[u8],
    mut i: usize,
    mut cp: u32,
    mut w: usize,
    out: &mut String,
    e: &mut Emit,
) -> usize {
    let n = bytes.len();
    loop {
        if HANGUL.contains(&cp) {
            // Hangul syllable run (checked BEFORE the trie gather): arithmetic decomposition, jamo are
            // 3-byte starters. A TIGHT inner loop eats the whole syllable run — direct 3-byte decode +
            // lead-range test per step, none of the general continuation machinery — because Korean is
            // the densest set-run script and per-syllable glue is the difference vs xxUTF.
            let mut scp = cp;
            loop {
                let s = scp - 0xAC00;
                let t = s % 28;
                let mut buf = [0u8; 16];
                for (slot, j) in [0x1100 + s / 588, 0x1161 + (s % 588) / 28, 0x11A7 + t]
                    .into_iter()
                    .enumerate()
                {
                    buf[slot * 3] = 0xE0 | (j >> 12) as u8;
                    buf[slot * 3 + 1] = 0x80 | ((j >> 6) & 0x3F) as u8;
                    buf[slot * 3 + 2] = 0x80 | (j & 0x3F) as u8;
                }
                // SAFETY: 16-byte stack buffer; capacity reserved (+16).
                unsafe { raw_extend(out, buf.as_ptr(), 16, if t != 0 { 9 } else { 6 }) };
                i += 3;
                if i + 3 <= n && (0xEA..=0xED).contains(&bytes[i]) {
                    let c2 = (((bytes[i] & 0x0F) as u32) << 12)
                        | (((bytes[i + 1] & 0x3F) as u32) << 6)
                        | ((bytes[i + 2] & 0x3F) as u32);
                    if HANGUL.contains(&c2) {
                        scp = c2;
                        continue;
                    }
                }
                break;
            }
            e.last_ccc = 0;
            e.run_start = out.len();
            // fall through to the generic continuation (the next char is NOT a syllable)
            w = 0; // already advanced past the run
        } else {
            let (off, blen) = D::byte_entry(cp);
            // SAFETY: blob offsets point ≥ 3 in (headers precede the bytes); 16-byte zero tail pad.
            let first = unsafe { *D::BYTE_BLOB.get_unchecked(off - 3) };
            if blen > 16 || (first != 0 && first < e.last_ccc) {
                // oversized decomposition or out-of-order mark: the scalar path reorders correctly
                decompose_char::<D>(cp, out, e);
            } else {
                let pos = out.len();
                // SAFETY: fixed 16-byte copy from the padded blob; capacity reserved.
                unsafe { raw_extend(out, D::BYTE_BLOB.as_ptr().add(off), 16, blen) };
                let (last, mark_off) = unsafe {
                    (*D::BYTE_BLOB.get_unchecked(off - 2), *D::BYTE_BLOB.get_unchecked(off - 1) as usize)
                };
                e.last_ccc = last;
                if mark_off > 0 {
                    // contains a starter: the current mark run begins just past the last one
                    e.run_start = pos + mark_off;
                } // else pure marks: the run continues, run_start stays
            }
        }
        i += w;
        // Continue through SHORT stable gaps: real text alternates letter↔mark (Hebrew niqqud, Hindi
        // virama, Greek accents) and syllable↔space (Korean) — bailing to the driver per single stable
        // char costs a `push_str` + `next_set` round-trip each time. A stable char SANDWICHED between
        // set chars is emitted inline (exact 1–4 byte copy); two stable chars in a row = a real stable
        // run, where the driver's bulk skip+memcpy wins — bail.
        loop {
            if i >= n {
                return i;
            }
            // two ASCII bytes ahead = a real ASCII run: bail to the driver's fused skip+copy without
            // decoding anything (the common exit in Latin text — one byte compare each).
            if bytes[i] < 0x80 && (i + 1 >= n || bytes[i + 1] < 0x80) {
                return i;
            }
            let (ncp, nw) = decode_cp(bytes, i);
            // Hangul range first: Korean set-runs are the densest, and the range test on the decoded
            // cp saves the two bitset loads per syllable that `bit_set` would spend.
            if HANGUL.contains(&ncp) || bit_set::<D>(ncp) {
                (cp, w) = (ncp, nw);
                break; // next unstable char: emit it
            }
            let j = i + nw;
            if j >= n {
                return i; // stable tail: driver copies it
            }
            let (cp2, w2) = decode_cp(bytes, j);
            if !bit_set::<D>(cp2) {
                return i; // two stable in a row: bail to the bulk skip
            }
            // SAFETY: exact nw-byte in-bounds copy (nw ≤ 4); capacity reserved.
            unsafe { raw_extend(out, bytes.as_ptr().add(i), nw, nw) };
            e.last_ccc = 0;
            e.run_start = out.len();
            (cp, w) = (cp2, w2);
            i = j;
            break;
        }
    }
}

/// The `next_set`-driven owned path (aarch64): stable regions found by `next_set` are memcpy'd once,
/// each unstable run goes through [`emit_set`]. The right strategy for ASCII-dominant and CJK-block
/// text, where [`decompose_tagged`]'s whole-buffer classify costs more than it saves (see [`decompose`]).
#[cfg(target_arch = "aarch64")]
fn decompose_owned<D: Decomp>(bytes: &[u8], out: &mut String, e: &mut Emit, mut i: usize) {
    use std::arch::aarch64::*;
    let n = bytes.len();
    while i < n {
        if bytes[i] < 0x80 {
            // fused ASCII scan+copy: the `vld1q` needed for the all-ASCII test IS the copy source, so
            // ASCII-dominant text (French & friends) writes through instead of scan-then-memcpy. The
            // store is UNCONDITIONAL (over-store rides the +16 capacity slack) and a movemask jumps
            // straight to the first non-ASCII byte — no per-byte tail before every accent.
            // SAFETY: loads/stores gated by `i + 16 <= n` and reserved capacity; exact scalar tail.
            unsafe {
                const POW: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
                let powv = vld1q_u8(POW.as_ptr());
                let v = out.as_mut_vec();
                let mut len = v.len();
                while i + 16 <= n {
                    let x = vld1q_u8(bytes.as_ptr().add(i));
                    vst1q_u8(v.as_mut_ptr().add(len), x);
                    if vmaxvq_u8(x) < 0x80 {
                        len += 16;
                        i += 16;
                        continue;
                    }
                    // first non-ASCII lane: keep the ASCII prefix of this chunk, stop there
                    let hb = vandq_u8(vcgeq_u8(x, vdupq_n_u8(0x80)), powv);
                    let mm = (vaddv_u8(vget_low_u8(hb)) as u16)
                        | ((vaddv_u8(vget_high_u8(hb)) as u16) << 8);
                    let k = mm.trailing_zeros() as usize;
                    len += k;
                    i += k;
                    break;
                }
                while i < n && bytes[i] < 0x80 {
                    *v.as_mut_ptr().add(len) = bytes[i];
                    len += 1;
                    i += 1;
                }
                v.set_len(len);
            }
            e.last_ccc = 0;
            e.run_start = out.len();
            continue;
        }
        // 3-byte leads (CJK/kana — the dense-emit scripts): the write-through skip scans AND copies in
        // one pass, handing over the decoded SET char it stops at. Handles bytes[i] itself being SET
        // (lane 0 stops immediately with cp forwarded).
        if (0xE0..0xF0).contains(&bytes[i]) && i + 48 <= n {
            let before = out.len();
            let (ni, ncp) = skip_clear_3byte_copy::<D>(bytes, i, out);
            if out.len() > before {
                e.last_ccc = 0;
                e.run_start = out.len();
            }
            if ncp != 0 {
                i = emit_set::<D>(bytes, ni, ncp, 3, out, e);
                continue;
            }
            if ni > i {
                i = ni;
                continue;
            }
        }
        // inline dispatch: a SET char right after an ASCII run (the à/é/ç case in Latin text) goes
        // straight to the emit — no `next_set` call-and-return per accent.
        let (cp0, w0) = decode_cp(bytes, i);
        if bit_set::<D>(cp0) {
            i = emit_set::<D>(bytes, i, cp0, w0, out, e);
            continue;
        }
        let (ns, cp, w) = next_set::<D>(bytes, i);
        if ns > i {
            // stable region: one callless copy; it ends the previous combining sequence (ccc-0 starters)
            // SAFETY: exact in-bounds verbatim span; capacity reserved.
            unsafe { copy_small(out, bytes.as_ptr().add(i), ns - i) };
            e.last_ccc = 0;
            e.run_start = out.len();
        }
        if ns == n {
            return;
        }
        i = emit_set::<D>(bytes, ns, cp, w, out, e);
    }
}

/// Exact ccc RANK of the odd (`0x3D`-tagged) mark at `pos` — decode + entry + ccc→rank map. Rare:
/// only compat-decomposing identity marks take this path.
#[cfg(target_arch = "aarch64")]
#[cold]
fn rank_probe(bytes: &[u8], pos: usize) -> u8 {
    let (cp, _) = decode_cp(bytes, pos);
    let ccc = (Nfd::entries(cp)[0] >> 24) as u8;
    ccc_rank_map()[&ccc]
}

/// First index ≥ `i` whose check tag is DECOMPOSE-relevant — nonzero, not the composition-only `0x40`
/// (a Maybe STARTER: stable under decomposition), and a real tag (bit 7 excludes the CONT/MB
/// sentinels) — or `tags.len()`. The whole-buffer scan the
/// per-word `next_set` machinery can't match: 16 tags per `vld1q`, indifferent to char-width mixing,
/// so short words / punctuation cost nothing extra.
#[cfg(target_arch = "aarch64")]
#[inline]
fn next_check_cand(tags: &[u8], mut i: usize) -> usize {
    use std::arch::aarch64::*;
    let n = tags.len();
    // SAFETY: every load is gated by `i + 16 <= n`; NEON loads are alignment-free.
    unsafe {
        let comp = vdupq_n_u8(0x40);
        let hb = vdupq_n_u8(0x80);
        while i + 16 <= n {
            let v = vld1q_u8(tags.as_ptr().add(i));
            // candidate: t != 0, t != 0x40 (composition-only Maybe starter), bit 7 clear (sentinels)
            let nz = vtstq_u8(v, v); // (t & t) != 0 ⇔ t != 0
            let hit = vbicq_u8(vbicq_u8(nz, vceqq_u8(v, comp)), vtstq_u8(v, hb));
            if vmaxvq_u8(hit) != 0 {
                let mut h = [0u8; 16];
                vst1q_u8(h.as_mut_ptr(), hit);
                for (k, &x) in h.iter().enumerate() {
                    if x != 0 {
                        return i + k;
                    }
                }
            }
            i += 16;
        }
    }
    while i < n {
        let t = tags[i];
        if t != 0 && t != 0x40 && t & 0x80 == 0 {
            return i;
        }
        i += 1;
    }
    n
}

/// Whole-buffer decompose (aarch64): ONE `norm_classify` pass tags every byte, then both phases are
/// candidate-driven — the check walks tag hits verifying canonical order, and on failure the owned
/// rebuild memcpys the stable gaps between hits and routes each confirmed unstable run through
/// [`emit_set`] (blob copy / arithmetic Hangul). This is the xxUTF-style structure: no per-word SIMD
/// re-entry, no per-char scan cost on stable text of ANY width mix — the scan is `vld1q`+`vtst` flat.
#[cfg(target_arch = "aarch64")]
fn decompose_tagged<'a, D: Decomp>(input: &'a str) -> Cow<'a, str> {
    let bytes = input.as_bytes();
    let n = bytes.len();
    TAG_SCRATCH.with(|sc| {
        let tags = &mut *sc.borrow_mut();
        tags.resize(n, 0); // grow-only in steady state; classify overwrites [0..n)
        crate::classify::classify_with::<0x81, 0x80, { crate::classify::NO_CJK }>(
            bytes,
            tags,
            &crate::nfd_check_tables::NFD_CHECK_TABLES,
        );
        let tags = &tags[..n];

        // ── check phase: PURE tag arithmetic — no decode, no table loads. `t >= BREAK_MIN` means the
        // char changes under this form; otherwise `t & 0x3F` is its ccc RANK (order-preserving), so
        // canonical order between adjacent marks is one compare. Gaps and rank-0 chars close the
        // combining sequence.
        let mut i = 0;
        let mut prev_rank = 0u8;
        let brk = loop {
            let j = next_check_cand(tags, i);
            if j == n {
                return Cow::Borrowed(input);
            }
            let t = tags[j];
            if D::tag_breaks(t) {
                break j; // decomposition changes it (Hangul included)
            }
            if j > i {
                prev_rank = 0; // a stable gap closed the combining sequence
            }
            let r = match t & 0x3F {
                0x3C => 0,                       // compat starter (NFD check): ccc-0, closes the sequence
                0x3D => rank_probe(bytes, j),    // odd compat mark: rare exact-rank probe
                r => r,                          // plain / Maybe mark: rank rides in the low bits
            };
            if r == 0 {
                prev_rank = 0;
            } else {
                if r < prev_rank {
                    break j; // reorderable mark out of canonical order
                }
                prev_rank = r;
            }
            // width from the lead byte (candidates are never ASCII: marks/decomposables are ≥ U+0300)
            let b = bytes[j];
            i = j + if b < 0xE0 {
                2
            } else if b < 0xF0 {
                3
            } else {
                4
            };
        };

        // ── owned rebuild: resume at the stable char before `brk` (its re-processing resets `Emit`) ──
        let mut s = brk;
        while s > 0 {
            let mut p = s - 1;
            while p > 0 && bytes[p] & 0xC0 == 0x80 {
                p -= 1;
            }
            s = p;
            let (cp, _) = decode_cp(bytes, p);
            if !bit_set::<D>(cp) {
                break;
            }
        }
        let mut out = String::with_capacity(n.saturating_mul(D::MAX_EXPAND) + 48);
        out.push_str(&input[..s]);
        let mut e = Emit { last_ccc: 0, run_start: out.len() };
        // `gap` = start of pending uncopied stable text. FALSE candidates (spacing marks etc. — dense in
        // Thai/Devanagari) stay inside the gap: they cost one decode+probe, never a copy of their own.
        // The owned rebuild is verify-as-you-go, exactly like the check: IN-ORDER identity marks stay
        // inside the gap (their blob is themselves — verbatim copy is the same bytes), so text after a
        // lone breaking char (a stray accented Latin word in a Thai doc, say) is still one big memcpy.
        // Only breaking chars and out-of-order marks pay `emit_set` — and when one directly follows
        // in-gap marks, that trailing mark run is re-emitted through `emit_set` first so the `Emit`
        // chain state (`last_ccc`/`run_start`) is exact for the reorder logic.
        let (mut gap, mut i) = (s, s);
        let mut prev_rank = 0u8;
        let mut run_begin = usize::MAX; // first mark of the current in-gap mark run (adjacent chain)
        while i < n {
            let j = next_check_cand(tags, i);
            if j == n {
                break;
            }
            let t = tags[j];
            if j > i {
                // ≥1 stable char between candidates: the combining sequence closed inside the gap
                prev_rank = 0;
                run_begin = usize::MAX;
            }
            if !D::tag_breaks(t) {
                let b = bytes[j];
                let w = if b < 0xE0 { 2 } else if b < 0xF0 { 3 } else { 4 };
                // 0x3D (odd compat mark) falls through to the emit path — exactness over speed, rare
                if t & 0x3F != 0x3D {
                    let r = if t & 0x3F == 0x3C { 0 } else { t & 0x3F };
                    if r == 0 {
                        // stable starter under this form (e.g. compat-only char under NFD): stays in gap
                        prev_rank = 0;
                        run_begin = usize::MAX;
                        i = j + w;
                        continue;
                    }
                    if r >= prev_rank {
                        // in-order identity mark: stays in the gap
                        if prev_rank == 0 {
                            run_begin = j;
                        }
                        prev_rank = r;
                        i = j + w;
                        continue;
                    }
                }
                // out-of-order or odd mark: real work below
            }
            // Breaking char or out-of-order mark. Re-emit any in-gap mark run adjacent to it so the
            // Emit state is exact; otherwise start emitting at j itself.
            let start = if run_begin != usize::MAX { run_begin } else { j };
            if start > gap {
                // SAFETY: exact in-bounds verbatim span; capacity reserved.
                unsafe { copy_small(&mut out, bytes.as_ptr().add(gap), start - gap) };
                e.last_ccc = 0;
                e.run_start = out.len();
            }
            let (cp, w) = decode_cp(bytes, start);
            debug_assert!(bit_set::<D>(cp), "check-tag candidate must be in the bitset");
            i = emit_set::<D>(bytes, start, cp, w, &mut out, &mut e);
            gap = i;
            prev_rank = 0; // emit_set consumed the whole adjacent set-run: a gap follows by construction
            run_begin = usize::MAX;
        }
        if n > gap {
            // SAFETY: exact in-bounds verbatim tail; capacity reserved.
            unsafe { copy_small(&mut out, bytes.as_ptr().add(gap), n - gap) };
        }
        Cow::Owned(out)
    })
}

/// Decompose `input` under form `D`. `Cow::Borrowed` when already in that form (the common case — see
/// [`normalized_prefix`]), else owned. The owned path copies the already-verified prefix wholesale
/// (no re-scan), rewinds to the nearest stable char (whose processing trivially re-establishes the
/// `Emit` state), then runs the whole-buffer kernel: on aarch64 the chunked SIMD [`decompose_owned`]
/// (uniform-width `vld2q`/`vld3q` decode, bulk stable-run memcpy, 16-byte blob emits, arithmetic
/// Hangul); elsewhere the portable `next_set` + [`decompose_char`] loop.
/// Streaming check for 3-byte single-block scripts (Thai, Devanagari, Lao, …): NO tag buffer — per
/// uniform run the block's own 128-cp check-tag table (the same `fast3` data the classify engine uses)
/// is probed IN REGISTERS via `vqtbl4`, with break detection and the canonical-order rank compare as
/// vector ops. Anything irregular falls out to a scalar per-char verify (`Tables::classify_char` with
/// the same tag semantics), which then re-enters the vector run — the vector is a pure accelerator,
/// the scalar referee keeps it byte-exact.
#[cfg(target_arch = "aarch64")]
fn streaming3_prefix<D: Decomp>(bytes: &[u8]) -> usize {
    use std::arch::aarch64::*;
    let t = &crate::nfd_check_tables::NFD_CHECK_TABLES;
    let n = bytes.len();
    let mut i = 0;
    let mut prev_rank = 0u8;
    // Block-cached vector state: table registers survive ASCII holes and scalar bails, so re-entering
    // the vector run for the same block (the common case — Thai text is one block) costs nothing.
    let mut cur_blk = usize::MAX;
    let mut uni = 0u8;
    // SAFETY: all loads gated by explicit bounds; NEON ops are value-level.
    unsafe {
        let mut leadv = vdupq_n_u8(0);
        let mut pairv = vdupq_n_u8(0);
        let mut lo4 = vld1q_u8_x4([0u8; 64].as_ptr());
        let mut hi4 = lo4;
        while i < n {
            let b = bytes[i];
            if b < 0x80 {
                prev_rank = 0;
                i = skip_ascii(bytes, i);
                continue;
            }
            'vector: {
                if (0xE0..0xF0).contains(&b) && i + 48 <= n {
                    let blk = ((b - 0xE0) as usize) * 32 + ((bytes[i + 1] >> 1) & 0x1F) as usize;
                    if blk != cur_blk {
                        // Only rebuild the cached tables for a SUSTAINED block change (next char in the
                        // same new block). Lone foreign chars — Thai text is peppered with U+200B
                        // zero-width spaces — go to the scalar referee with the cache intact.
                        let sustained = i + 6 <= n
                            && bytes[i + 3] == b
                            && (bytes[i + 4] >> 1) == (bytes[i + 1] >> 1);
                        if !sustained {
                            break 'vector;
                        }
                        cur_blk = blk;
                        uni = t.fast3_uni[blk];
                        leadv = vdupq_n_u8(b);
                        pairv = vdupq_n_u8(bytes[i + 1] >> 1);
                        if uni == 0xFF {
                            let (l, h) = &t.fast3_mixed[t.fast3_slot[blk] as usize];
                            lo4 = vld1q_u8_x4(l.as_ptr());
                            hi4 = vld1q_u8_x4(h.as_ptr());
                        }
                    }
                    while i + 48 <= n {
                        let x = vld3q_u8(bytes.as_ptr().add(i));
                        let same =
                            vandq_u8(vceqq_u8(x.0, leadv), vceqq_u8(vshrq_n_u8::<1>(x.1), pairv));
                        let idx7 = vorrq_u8(
                            vshlq_n_u8::<6>(vandq_u8(x.1, vdupq_n_u8(1))),
                            vandq_u8(x.2, vdupq_n_u8(0x3F)),
                        );
                        let tag = if uni != 0xFF {
                            vdupq_n_u8(uni)
                        } else {
                            vorrq_u8(
                                vqtbl4q_u8(lo4, idx7),
                                vqtbl4q_u8(hi4, vsubq_u8(idx7, vdupq_n_u8(64))),
                            )
                        };
                        // irregular: any form-breaking tag (0x3C/0x3D/0x7D/0x7E — the scalar referee
                        // decides precisely); canonical order: prev > cur with both ranks nonzero
                        let r = vandq_u8(tag, vdupq_n_u8(0x3F));
                        let irregular = vorrq_u8(
                            vcgeq_u8(tag, vdupq_n_u8(0x7D)),
                            vcgeq_u8(r, vdupq_n_u8(0x3C)),
                        );
                        let prevr = vextq_u8::<15>(vdupq_n_u8(prev_rank), r);
                        let viol = vandq_u8(
                            vcgtq_u8(prevr, r),
                            vandq_u8(vtstq_u8(r, r), vtstq_u8(prevr, prevr)),
                        );
                        let bad = vorrq_u8(vorrq_u8(vmvnq_u8(same), irregular), viol);
                        if vmaxvq_u8(bad) == 0 {
                            prev_rank = vgetq_lane_u8::<15>(r);
                            i += 48;
                            continue;
                        }
                        let (mut m, mut rr) = ([0u8; 16], [0u8; 16]);
                        vst1q_u8(m.as_mut_ptr(), bad);
                        vst1q_u8(rr.as_mut_ptr(), r);
                        for (l, &v) in m.iter().enumerate() {
                            if v != 0 {
                                if l > 0 {
                                    prev_rank = rr[l - 1];
                                }
                                i += l * 3;
                                break 'vector; // scalar referee takes the bad lane
                            }
                        }
                    }
                    if i >= n {
                        return n;
                    }
                }
            }
            // scalar referee: one char via the SAME tag semantics
            let tag = t.classify_char(bytes, i);
            if D::tag_breaks(tag) {
                return i;
            }
            let r = match tag & 0x3F {
                0x3C => 0,
                0x3D => rank_probe(bytes, i),
                r => r,
            };
            if r == 0 {
                prev_rank = 0;
            } else {
                if r < prev_rank {
                    return i;
                }
                prev_rank = r;
            }
            let w = if bytes[i] < 0x80 {
                1 // a vector bail can land on ASCII (word spaces): the referee must not stride over it
            } else if bytes[i] < 0xE0 {
                2
            } else if bytes[i] < 0xF0 {
                3
            } else {
                4
            };
            i += w;
        }
    }
    n
}

/// Owned twin of [`streaming3_prefix`]: the same in-register run verification, but verify-as-you-go —
/// verified spans stay in a deferred gap (ONE memcpy each), and only form-breaking chars pay a rewind
/// (back over the adjacent in-gap marks to their starter, restoring exact `Emit` state) + [`emit_set`].
#[cfg(target_arch = "aarch64")]
fn streaming3_owned<D: Decomp>(input: &str, out: &mut String, e: &mut Emit, s: usize) {
    use std::arch::aarch64::*;
    let bytes = input.as_bytes();
    let t = &crate::nfd_check_tables::NFD_CHECK_TABLES;
    let n = bytes.len();
    let (mut gap, mut i) = (s, s);
    let mut prev_rank = 0u8;
    let mut cur_blk = usize::MAX;
    let mut uni = 0u8;
    // SAFETY: all loads gated by explicit bounds; NEON ops are value-level.
    unsafe {
        let mut leadv = vdupq_n_u8(0);
        let mut pairv = vdupq_n_u8(0);
        let mut lo4 = vld1q_u8_x4([0u8; 64].as_ptr());
        let mut hi4 = lo4;
        'outer: while i < n {
            let b = bytes[i];
            if b < 0x80 {
                prev_rank = 0;
                i = skip_ascii(bytes, i);
                continue;
            }
            'vector: {
                if (0xE0..0xF0).contains(&b) && i + 48 <= n {
                    let blk = ((b - 0xE0) as usize) * 32 + ((bytes[i + 1] >> 1) & 0x1F) as usize;
                    if blk != cur_blk {
                        let sustained = i + 6 <= n
                            && bytes[i + 3] == b
                            && (bytes[i + 4] >> 1) == (bytes[i + 1] >> 1);
                        if !sustained {
                            break 'vector;
                        }
                        cur_blk = blk;
                        uni = t.fast3_uni[blk];
                        leadv = vdupq_n_u8(b);
                        pairv = vdupq_n_u8(bytes[i + 1] >> 1);
                        if uni == 0xFF {
                            let (l, h) = &t.fast3_mixed[t.fast3_slot[blk] as usize];
                            lo4 = vld1q_u8_x4(l.as_ptr());
                            hi4 = vld1q_u8_x4(h.as_ptr());
                        }
                    }
                    while i + 48 <= n {
                        let x = vld3q_u8(bytes.as_ptr().add(i));
                        let same =
                            vandq_u8(vceqq_u8(x.0, leadv), vceqq_u8(vshrq_n_u8::<1>(x.1), pairv));
                        let idx7 = vorrq_u8(
                            vshlq_n_u8::<6>(vandq_u8(x.1, vdupq_n_u8(1))),
                            vandq_u8(x.2, vdupq_n_u8(0x3F)),
                        );
                        let tag = if uni != 0xFF {
                            vdupq_n_u8(uni)
                        } else {
                            vorrq_u8(
                                vqtbl4q_u8(lo4, idx7),
                                vqtbl4q_u8(hi4, vsubq_u8(idx7, vdupq_n_u8(64))),
                            )
                        };
                        let r = vandq_u8(tag, vdupq_n_u8(0x3F));
                        let irregular = vorrq_u8(
                            vcgeq_u8(tag, vdupq_n_u8(0x7D)),
                            vcgeq_u8(r, vdupq_n_u8(0x3C)),
                        );
                        let prevr = vextq_u8::<15>(vdupq_n_u8(prev_rank), r);
                        let viol = vandq_u8(
                            vcgtq_u8(prevr, r),
                            vandq_u8(vtstq_u8(r, r), vtstq_u8(prevr, prevr)),
                        );
                        let bad = vorrq_u8(vorrq_u8(vmvnq_u8(same), irregular), viol);
                        if vmaxvq_u8(bad) == 0 {
                            prev_rank = vgetq_lane_u8::<15>(r);
                            i += 48;
                            continue;
                        }
                        let (mut m, mut rr) = ([0u8; 16], [0u8; 16]);
                        vst1q_u8(m.as_mut_ptr(), bad);
                        vst1q_u8(rr.as_mut_ptr(), r);
                        for (l, &v) in m.iter().enumerate() {
                            if v != 0 {
                                if l > 0 {
                                    prev_rank = rr[l - 1];
                                }
                                i += l * 3;
                                break 'vector;
                            }
                        }
                    }
                    if i >= n {
                        break 'outer;
                    }
                }
            }
            if bytes[i] < 0x80 {
                continue; // vector bailed onto ASCII: loop top handles it
            }
            // scalar referee: verified chars stay in the gap; breaking/out-of-order chars emit
            let tag = t.classify_char(bytes, i);
            let breaks = D::tag_breaks(tag);
            let r = if breaks {
                0
            } else {
                match tag & 0x3F {
                    0x3C => 0,
                    0x3D => rank_probe(bytes, i),
                    r => r,
                }
            };
            if !breaks && !(r != 0 && r < prev_rank) {
                prev_rank = if r == 0 { 0 } else { r };
                let w = if bytes[i] < 0xE0 { 2 } else if bytes[i] < 0xF0 { 3 } else { 4 };
                i += w;
                continue;
            }
            // rewind over the adjacent in-gap marks to their starter so `Emit` state is exact
            let mut st = i;
            while st > gap {
                let mut p = st - 1;
                while p > gap && bytes[p] & 0xC0 == 0x80 {
                    p -= 1;
                }
                let (pcp, _) = decode_cp(bytes, p);
                if !bit_set::<D>(pcp) {
                    break;
                }
                st = p;
            }
            if st > gap {
                // (in the enclosing unsafe scope) exact in-bounds verbatim span; capacity reserved.
                copy_small(out, bytes.as_ptr().add(gap), st - gap);
                e.last_ccc = 0;
                e.run_start = out.len();
            }
            let (cp, w) = decode_cp(bytes, st);
            i = if bit_set::<D>(cp) {
                emit_set::<D>(bytes, st, cp, w, out, e)
            } else {
                // breaking tag but stable in D's bitset can't happen (tags derive from the same
                // tables); defensive: copy verbatim
                copy_small(out, bytes.as_ptr().add(st), w);
                st + w
            };
            gap = i;
            prev_rank = 0;
            cur_blk = usize::MAX; // emit may have crossed blocks; re-resolve lazily
        }
    }
    if n > gap {
        // SAFETY: exact in-bounds verbatim tail; capacity reserved.
        unsafe { copy_small(out, bytes.as_ptr().add(gap), n - gap) };
    }
}

/// Streaming COMPOSE quick-check for 3-byte single-block scripts (Thai/Lao): the same in-register
/// vector core as [`streaming3_prefix`] with compose predicates — `0x7E` (decomposing but
/// composition-stable) and `0x3C` (compat starter, stable under NFC) count as rank-0 starters;
/// QC-Maybe (`0x40`), `0x7D`, `0x3D`, and (for the K form) `0x3C` bail as relevant. Returns
/// `bytes.len()` iff the input is already in the composed form — the caller borrows; on the first
/// relevant char it returns its position and the caller falls back to the full compose machinery.
#[cfg(target_arch = "aarch64")]
fn streaming3_compose_ok<D: Decomp>(bytes: &[u8]) -> bool {
    use std::arch::aarch64::*;
    let t = &crate::nfd_check_tables::NFD_CHECK_TABLES;
    let n = bytes.len();
    let mut i = 0;
    let mut prev_rank = 0u8;
    let mut cur_blk = usize::MAX;
    let mut uni = 0u8;
    let kform = D::tag_breaks(0x3C);
    // SAFETY: all loads gated by explicit bounds; NEON ops are value-level.
    unsafe {
        let mut leadv = vdupq_n_u8(0);
        let mut pairv = vdupq_n_u8(0);
        let mut lo4 = vld1q_u8_x4([0u8; 64].as_ptr());
        let mut hi4 = lo4;
        while i < n {
            let b = bytes[i];
            if b < 0x80 {
                prev_rank = 0;
                i = skip_ascii(bytes, i);
                continue;
            }
            'vector: {
                if (0xE0..0xF0).contains(&b) && i + 48 <= n {
                    let blk = ((b - 0xE0) as usize) * 32 + ((bytes[i + 1] >> 1) & 0x1F) as usize;
                    if blk != cur_blk {
                        let sustained = i + 6 <= n
                            && bytes[i + 3] == b
                            && (bytes[i + 4] >> 1) == (bytes[i + 1] >> 1);
                        if !sustained {
                            break 'vector;
                        }
                        cur_blk = blk;
                        uni = t.fast3_uni[blk];
                        leadv = vdupq_n_u8(b);
                        pairv = vdupq_n_u8(bytes[i + 1] >> 1);
                        if uni == 0xFF {
                            let (l, h) = &t.fast3_mixed[t.fast3_slot[blk] as usize];
                            lo4 = vld1q_u8_x4(l.as_ptr());
                            hi4 = vld1q_u8_x4(h.as_ptr());
                        }
                    }
                    while i + 48 <= n {
                        let x = vld3q_u8(bytes.as_ptr().add(i));
                        let same =
                            vandq_u8(vceqq_u8(x.0, leadv), vceqq_u8(vshrq_n_u8::<1>(x.1), pairv));
                        let idx7 = vorrq_u8(
                            vshlq_n_u8::<6>(vandq_u8(x.1, vdupq_n_u8(1))),
                            vandq_u8(x.2, vdupq_n_u8(0x3F)),
                        );
                        let tag = if uni != 0xFF {
                            vdupq_n_u8(uni)
                        } else {
                            vorrq_u8(
                                vqtbl4q_u8(lo4, idx7),
                                vqtbl4q_u8(hi4, vsubq_u8(idx7, vdupq_n_u8(64))),
                            )
                        };
                        // compose-relevant: Maybe flag (0x40..=0x7B — NOT the 0x7D/0x7E break values,
                        // which also carry bit 6!), 0x7D, 0x3D — plus 0x3C under the K form
                        let maybe =
                            vbicq_u8(vtstq_u8(tag, vdupq_n_u8(0x40)), vcgeq_u8(tag, vdupq_n_u8(0x7D)));
                        let mut relevant = vorrq_u8(
                            maybe,
                            vorrq_u8(
                                vceqq_u8(tag, vdupq_n_u8(0x7D)),
                                vceqq_u8(tag, vdupq_n_u8(0x3D)),
                            ),
                        );
                        if kform {
                            relevant = vorrq_u8(relevant, vceqq_u8(tag, vdupq_n_u8(0x3C)));
                        }
                        // rank for the order check: 0x3C / 0x7E are ccc-0 starters → rank 0
                        let starters = vorrq_u8(
                            vceqq_u8(tag, vdupq_n_u8(0x3C)),
                            vceqq_u8(tag, vdupq_n_u8(0x7E)),
                        );
                        let r = vbicq_u8(vandq_u8(tag, vdupq_n_u8(0x3F)), starters);
                        let prevr = vextq_u8::<15>(vdupq_n_u8(prev_rank), r);
                        let viol = vandq_u8(
                            vcgtq_u8(prevr, r),
                            vandq_u8(vtstq_u8(r, r), vtstq_u8(prevr, prevr)),
                        );
                        let bad = vorrq_u8(vorrq_u8(vmvnq_u8(same), relevant), viol);
                        if vmaxvq_u8(bad) == 0 {
                            prev_rank = vgetq_lane_u8::<15>(r);
                            i += 48;
                            continue;
                        }
                        let (mut m, mut rr) = ([0u8; 16], [0u8; 16]);
                        vst1q_u8(m.as_mut_ptr(), bad);
                        vst1q_u8(rr.as_mut_ptr(), r);
                        for (l, &v) in m.iter().enumerate() {
                            if v != 0 {
                                if l > 0 {
                                    prev_rank = rr[l - 1];
                                }
                                i += l * 3;
                                break 'vector;
                            }
                        }
                    }
                    if i >= n {
                        return true;
                    }
                }
            }
            if bytes[i] < 0x80 {
                continue; // vector bailed onto ASCII
            }
            // scalar referee under compose semantics
            let tag = t.classify_char(bytes, i);
            let relevant = (tag & 0x40 != 0 && tag < 0x7D)
                || tag == 0x7D
                || tag == 0x3D
                || (kform && tag == 0x3C);
            if relevant {
                return false; // fall back to the full compose machinery
            }
            let r = if tag == 0x3C || tag == 0x7E { 0 } else { tag & 0x3F };
            if r == 0 {
                prev_rank = 0;
            } else {
                if r < prev_rank {
                    return false; // out of order: recompose needed
                }
                prev_rank = r;
            }
            let w = if bytes[i] < 0x80 {
                1
            } else if bytes[i] < 0xE0 {
                2
            } else if bytes[i] < 0xF0 {
                3
            } else {
                4
            };
            i += w;
        }
    }
    true
}

fn decompose<'a, D: Decomp>(input: &'a str) -> Cow<'a, str> {
    // aarch64: two strategies, dispatched by a ~64-byte content sample. The whole-buffer TAG kernel
    // (norm-classify once + candidate scans) wins on marked scripts of any width mix (Cyrillic, Hebrew,
    // Arabic, Greek, Devanagari, Thai) — flat scan cost, no per-word SIMD re-entry. The NEXT_SET kernel
    // wins where the classify pass itself is the cost: ASCII-dominant text (skip_ascii is near-free) and
    // CJK-block text (Han/Hangul/kana have no uniform norm tag, so classify pays the block-peel loop).
    // Both are byte-exact; the sample only picks the faster one.
    #[cfg(target_arch = "aarch64")]
    {
        let bytes = input.as_bytes();
        let n = bytes.len();
        let a0 = skip_ascii(bytes, 0);
        if a0 == n {
            return Cow::Borrowed(input); // pure ASCII is NFD/NFKD-stable
        }
        if n >= 256 {
            // 3-byte non-CJK scripts (leads E0..E2: Thai/Indic/Lao/…): the streaming in-register
            // check — no tag buffer, the block's fast3 table probed in registers per 16-char chunk.
            let span = n - a0;
            let (mut e0c, mut tot) = (0usize, 0usize);
            for k in 0..4 {
                let off = a0 + span * k / 4;
                let end = (off + 16).min(n);
                for (x, &bb) in bytes[off..end].iter().enumerate() {
                    tot += 1;
                    // Thai/Lao only (E0 B8..BB): spaceless scripts where vector runs stay long.
                    // Word-spaced E0-scripts (Devanagari etc.) do better on the tagged kernel.
                    e0c += usize::from(
                        bb == 0xE0
                            && bytes.get(off + x + 1).is_some_and(|&c| (0xB8..=0xBB).contains(&c)),
                    );
                }
            }
            if e0c * 4 >= tot {
                let brk = streaming3_prefix::<D>(bytes);
                if brk == n {
                    return Cow::Borrowed(input);
                }
                // owned: rewind to a stable boundary, then the streaming verify-as-you-go rebuild
                let mut s = brk;
                while s > 0 {
                    let mut p = s - 1;
                    while p > 0 && bytes[p] & 0xC0 == 0x80 {
                        p -= 1;
                    }
                    s = p;
                    let (cp, _) = decode_cp(bytes, p);
                    if !bit_set::<D>(cp) {
                        break;
                    }
                }
                let mut out = String::with_capacity(n.saturating_mul(D::MAX_EXPAND) + 48);
                out.push_str(&input[..s]);
                let mut e = Emit { last_ccc: 0, run_start: out.len() };
                streaming3_owned::<D>(input, &mut out, &mut e, s);
                return Cow::Owned(out);
            }
            if pick_tagged(bytes, a0) {
                return decompose_tagged::<D>(input);
            }
        }
        decompose_nextset::<D>(input)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        decompose_nextset::<D>(input)
    }
}

/// Sample ≤64 bytes across `[a0, n)` and vote: use the tag kernel unless the text is CJK-block-heavy
/// (norm-classify block-peel cost) or ASCII-dominant (skip_ascii already near-free). Wrong votes only
/// cost speed, never correctness.
#[cfg(target_arch = "aarch64")]
#[inline]
fn pick_tagged(bytes: &[u8], a0: usize) -> bool {
    let n = bytes.len();
    let span = n - a0;
    let (mut blocky, mut ascii, mut tot) = (0usize, 0usize, 0usize);
    for k in 0..4 {
        let off = a0 + span * k / 4;
        let end = (off + 16).min(n);
        for (idx, &b) in bytes[off..end].iter().enumerate() {
            tot += 1;
            ascii += usize::from(b < 0x80);
            // "blocky" scripts where next_set's uniform-width skip beats the classify pass:
            // kana/Han/Hangul lead bytes (the norm classify has no uniform tag for those blocks).
            let _ = idx;
            blocky += usize::from((0xE3..=0xED).contains(&b));
        }
    }
    // pure 3-byte-block text has ~1 lead per 3 bytes (~tot/3 with ascii dilution): ≥ tot/4 is decisive
    !(blocky * 4 >= tot || ascii * 2 >= tot)
}

/// `next_set`-driven decompose: `normalized_prefix` for the borrow check, then the owned loop —
/// [`decompose_owned`] (blob/Hangul emits) on aarch64, the portable `decompose_char` loop elsewhere.
fn decompose_nextset<'a, D: Decomp>(input: &'a str) -> Cow<'a, str> {
    let bytes = input.as_bytes();
    let n = bytes.len();
    let brk = normalized_prefix::<D>(bytes);
    if brk == n {
        return Cow::Borrowed(input);
    }
    decompose_rebuild::<D>(input, brk)
}

/// The owned rebuild shared by every check strategy: rewind from the breaking char to a stable
/// boundary, copy the verified prefix wholesale, run the owned kernel from there.
fn decompose_rebuild<'a, D: Decomp>(input: &'a str, brk: usize) -> Cow<'a, str> {
    let bytes = input.as_bytes();
    let n = bytes.len();
    // Resume point: back up from the breaking char to the previous STABLE (bit-clear) char boundary.
    // Everything before it is copied verbatim; the stable char itself is re-processed first, and being a
    // ccc-0 starter its processing resets `Emit` — so the resumed state is exact without re-deriving
    // `last_ccc`/`run_start` from the copied text. (The chars between it and `brk` are in-order marks —
    // they didn't break the form — and re-processing them from a reset state emits them unchanged.)
    let mut s = brk;
    while s > 0 {
        let mut p = s - 1;
        while p > 0 && bytes[p] & 0xC0 == 0x80 {
            p -= 1;
        }
        s = p;
        let (cp, _) = decode_cp(bytes, p);
        if !bit_set::<D>(cp) {
            break;
        }
    }
    // Reserve the worst-case decomposed size up front (`n * MAX_EXPAND`, +16 so the kernel's fixed
    // 16-byte blob stores never touch unreserved memory) — the owned path never reallocs mid-build.
    let mut out = String::with_capacity(n.saturating_mul(D::MAX_EXPAND) + 48);
    out.push_str(&input[..s]);
    let mut e = Emit { last_ccc: 0, run_start: out.len() };
    #[cfg(target_arch = "aarch64")]
    decompose_owned::<D>(bytes, &mut out, &mut e, s);
    #[cfg(not(target_arch = "aarch64"))]
    {
        // portable owned path: `next_set` bulk-skips stable runs, `decompose_char` handles the rest
        let (mut i, mut cp, mut w) = {
            let (ns, ncp, nw) = next_set::<D>(bytes, s);
            out.push_str(&input[s..ns]);
            if ns == n {
                return Cow::Owned(out);
            }
            e.last_ccc = 0;
            e.run_start = out.len();
            (ns, ncp, nw)
        };
        loop {
            // (cp, w) is the already-decoded unstable char at i (from next_set) — never re-decoded
            decompose_char::<D>(cp, &mut out, &mut e);
            i += w;
            let (ns, ncp, nw) = next_set::<D>(bytes, i);
            if ns > i {
                // a stable run followed: it starts with a ccc-0 starter, closing the combining sequence
                out.push_str(&input[i..ns]);
                e.last_ccc = 0;
                e.run_start = out.len();
            }
            if ns == n {
                break;
            }
            (i, cp, w) = (ns, ncp, nw);
        }
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
/// Tag-driven compose (aarch64): ONE check-tag classify, then the scan decides NFC/NFKC relevance
/// with tag arithmetic — plain rank marks are ORDER-checked in place (QC=Yes: they can't compose, so
/// in-order means untouched), `0x3E`/`0x3F` (QC-Maybe composables) and out-of-order marks open a
/// recompose window, compat tags (`>= 0x40`) open one under NFKC, and `0x7E` (canonically decomposing)
/// is confirmed against the R bitset — with Hangul-syllable leads (`EA..=ED`) skipped tag-only, since
/// composed syllables are NFC-stable. Real marked text (Thai/Hindi/Hebrew niqqud) borrows at scan speed.
#[cfg(target_arch = "aarch64")]
fn compose_tagged<'a, D: Decomp, R: Bitset>(input: &'a str) -> Cow<'a, str> {
    let bytes = input.as_bytes();
    let n = bytes.len();
    TAG_SCRATCH.with(|sc| {
        let tags_buf = &mut *sc.borrow_mut();
        tags_buf.resize(n, 0);
        crate::classify::classify_with::<0x81, 0x80, { crate::classify::NO_CJK }>(
            bytes,
            tags_buf,
            &crate::nfd_check_tables::NFD_CHECK_TABLES,
        );
        let tags = &tags_buf[..n];
        let mut out = String::new(); // allocated lazily on the first window (borrow = no alloc)
        COMPOSE_SCRATCH.with(|cs| {
            let (buf, pending, marks) = &mut *cs.borrow_mut();
            let (mut gap, mut i) = (0usize, 0usize);
            let mut prev_rank = 0u8;
            while i < n {
                let j = next_check_cand_compose(tags, i);
                if j == n {
                    break;
                }
                let t = tags[j];
                if j > i {
                    prev_rank = 0; // a stable starter gap closed the combining sequence
                }
                let b = bytes[j];
                let w = if b < 0xE0 { 2 } else if b < 0xF0 { 3 } else { 4 };
                // does this char force a recompose window? — pure tag arithmetic, no decodes
                // (0x7E — decomposing but composition-stable é/ά/Hangul — never reaches here: the scan
                // itself skips it, and the byte gap it leaves resets the rank chain.)
                let relevant = if t == 0x7D {
                    true // decomposes AND composition-relevant (exclusions/QC≠Yes): window
                } else if t & 0x40 != 0 {
                    true // QC-Maybe composable (starter or mark): must recompose
                } else if t == 0x3C {
                    D::tag_breaks(0x3C) // compat starter: the K form recomposes, NFC skips it
                } else if t == 0x3D {
                    true // odd compat mark: conservative window (rare)
                } else {
                    // plain rank mark (QC=Yes — cannot compose): only ORDER can change it
                    if t < prev_rank {
                        true
                    } else {
                        prev_rank = t;
                        false
                    }
                };
                if !relevant {
                    if t == 0x3C {
                        prev_rank = 0; // ccc-0 starter closes the sequence
                    }
                    i = j + w;
                    continue;
                }
                if t == 0x3C {
                    // compat starter under the K form: if its full NFKC is baked as a neighbour-inert
                    // blob (all ccc-0, no back/forward composition possible), emit it directly — the
                    // window machinery (scratch buffers + COMPOSE searches) is overkill for `，`→`,`.
                    let (cp, _) = decode_cp(bytes, j);
                    // SAFETY: same trie shape as `entries`; cp < CAP for every 0x3C-tagged char.
                    let (off, blen) = unsafe {
                        let slot = *Nfkd::TRIE_INDEX.get_unchecked((cp >> 6) as usize) + (cp & 63);
                        let packed = *crate::nfd_byte_tables::NFKC_BYTE_DATA
                            .get_unchecked(slot as usize);
                        ((packed >> 8) as usize, (packed & 0xFF) as usize)
                    };
                    if (1..=16).contains(&blen) {
                        if out.capacity() == 0 {
                            out.reserve(n.saturating_mul(D::MAX_EXPAND));
                        }
                        out.push_str(&input[gap..j]);
                        out.push_str(unsafe {
                            std::str::from_utf8_unchecked(
                                &crate::nfd_byte_tables::NFKC_BYTE_BLOB[off..off + blen],
                            )
                        });
                        gap = j + w;
                        i = j + w;
                        prev_rank = 0;
                        continue;
                    }
                }
                // open a window: rewind over the WHOLE preceding combining cluster (any chars in D's
                // unstable bitset — marks and decomposing chars, e.g. in-order plain marks we skipped)
                // plus ONE starter: composition targets that starter, and everything between it and the
                // relevant char participates in the reorder+compose.
                let mut ws = j;
                while ws > gap {
                    let mut p = ws - 1;
                    while p > gap && bytes[p] & 0xC0 == 0x80 {
                        p -= 1;
                    }
                    ws = p;
                    if !bit_set::<D>(decode_cp(bytes, p).0) {
                        break; // reached the starter — include it and stop
                    }
                }
                let mut seg_end = j + w; // always consume the relevant char itself (progress even if ∉ R)
                while seg_end < n {
                    let (cp, cw) = decode_cp(bytes, seg_end);
                    if !bit_set::<R>(cp) {
                        break;
                    }
                    seg_end += cw;
                }
                if out.capacity() == 0 {
                    out.reserve(n.saturating_mul(D::MAX_EXPAND));
                }
                out.push_str(&input[gap..ws]);
                decompose_to_pairs::<D>(&input[ws..seg_end], buf, pending);
                compose_into(buf, &mut out, marks);
                gap = seg_end;
                i = seg_end;
                prev_rank = 0;
            }
            if gap == 0 {
                return Cow::Borrowed(input);
            }
            out.push_str(&input[gap..n]);
            Cow::Owned(out)
        })
    })
}

/// First index ≥ `i` whose check tag is COMPOSE-relevant: nonzero, not `0x7E` (a decomposing char
/// that is composition-STABLE — precomposed é/ά and Hangul syllables, the bulk of real text — its
/// ccc-0/QC=Yes nature means skipping it is exact: the byte gap it leaves resets the rank chain), and
/// a real tag (bit 7 excludes sentinels).
#[cfg(target_arch = "aarch64")]
#[inline]
fn next_check_cand_compose(tags: &[u8], mut i: usize) -> usize {
    use std::arch::aarch64::*;
    let n = tags.len();
    // SAFETY: every load is gated by `i + 16 <= n`; NEON loads are alignment-free.
    unsafe {
        let hb = vdupq_n_u8(0x80);
        let stable = vdupq_n_u8(0x7E);
        while i + 16 <= n {
            let v = vld1q_u8(tags.as_ptr().add(i));
            let hit =
                vbicq_u8(vbicq_u8(vtstq_u8(v, v), vceqq_u8(v, stable)), vtstq_u8(v, hb));
            if vmaxvq_u8(hit) != 0 {
                let mut h = [0u8; 16];
                vst1q_u8(h.as_mut_ptr(), hit);
                for (k, &x) in h.iter().enumerate() {
                    if x != 0 {
                        return i + k;
                    }
                }
            }
            i += 16;
        }
    }
    while i < n {
        let t = tags[i];
        if t != 0 && t != 0x7E && t & 0x80 == 0 {
            return i;
        }
        i += 1;
    }
    n
}

fn compose<'a, D: Decomp, R: Bitset>(input: &'a str) -> Cow<'a, str> {
    let bytes = input.as_bytes();
    let n = bytes.len();
    // aarch64: same strategy dispatch as decompose — the tag kernel for marked scripts; the bitset
    // scan for ASCII-dominant text (skip_ascii is near-free) and CJK-block text (the check classify
    // pays the block-peel there; the bitset scan skips Han/Hangul with the SIMD range kernels instead).
    #[cfg(target_arch = "aarch64")]
    {
        let a0 = skip_ascii(bytes, 0);
        if a0 == n {
            return Cow::Borrowed(input); // pure ASCII is NFC/NFKC-stable
        }
        if n >= 256 {
            // Thai/Lao: the streaming vector quick-check borrows at in-register speed; on failure the
            // tagged machinery below does the real work (double-checked only for non-normalized docs).
            let span = n - a0;
            let (mut e0c, mut tot) = (0usize, 0usize);
            for k in 0..4 {
                let off = a0 + span * k / 4;
                let end = (off + 16).min(n);
                for (x, &bb) in bytes[off..end].iter().enumerate() {
                    tot += 1;
                    e0c += usize::from(
                        bb == 0xE0
                            && bytes.get(off + x + 1).is_some_and(|&c| (0xB8..=0xBB).contains(&c)),
                    );
                }
            }
            if e0c * 4 >= tot && streaming3_compose_ok::<D>(bytes) {
                return Cow::Borrowed(input);
            }
            if pick_tagged(bytes, a0) {
                return compose_tagged::<D, R>(input);
            }
        }
    }
    // First composition-relevant char (ccc != 0, or QC != Yes). None ⇒ already in the target form.
    let mut rel = next_set::<R>(bytes, 0).0;
    if rel == n {
        return Cow::Borrowed(input);
    }
    // Multi-window recompose: copy every STABLE run verbatim (SIMD-skipped by `next_set::<R>`) and
    // decompose+recompose only the ACTIVE mark-clusters around each relevant char. A char not in R is
    // ccc-0 & QC=Yes — it neither composes backward nor is reached by a preceding composition, so it's a
    // safe segment boundary. Each active window is `[ws, seg_end)`: `ws` = the starter just before the
    // relevant char (composition's target), `seg_end` = the next not-in-R char. Byte-exact, and on
    // marked scripts (Cyrillic/Hebrew/Arabic) it copies the ~80% stable text instead of recomposing it.
    let mut out = String::with_capacity(n.saturating_mul(D::MAX_EXPAND));
    COMPOSE_SCRATCH.with(|sc| {
        let (buf, pending, marks) = &mut *sc.borrow_mut();
        let mut i = 0;
        loop {
            // Compat starter with a baked neighbour-inert NFKC blob (fullwidth punctuation etc. —
            // the dense case in CJK text under NFKC): emit the blob directly, no window scratch.
            if D::tag_breaks(0x3C) {
                let (cp, cw) = decode_cp(bytes, rel);
                if !HANGUL.contains(&cp) {
                    // SAFETY: same trie shape as `entries`; rel chars are unstable ⇒ cp < CAP.
                    let (off, blen) = unsafe {
                        let slot =
                            *Nfkd::TRIE_INDEX.get_unchecked((cp >> 6) as usize) + (cp & 63);
                        let packed = *crate::nfd_byte_tables::NFKC_BYTE_DATA
                            .get_unchecked(slot as usize);
                        ((packed >> 8) as usize, (packed & 0xFF) as usize)
                    };
                    if (1..=16).contains(&blen) {
                        // SAFETY: exact in-bounds verbatim span + padded blob; capacity reserved.
                        unsafe {
                            copy_small(&mut out, bytes.as_ptr().add(i), rel - i);
                            raw_extend(
                                &mut out,
                                crate::nfd_byte_tables::NFKC_BYTE_BLOB.as_ptr().add(off),
                                16,
                                blen,
                            );
                        }
                        i = rel + cw;
                        rel = next_set::<R>(bytes, i).0;
                        if rel == n {
                            out.push_str(&input[i..n]);
                            break;
                        }
                        continue;
                    }
                }
            }
            // back up one codepoint from `rel` to its starter (never below `i`), copy the stable `[i, ws)`
            let mut ws = rel;
            while ws > i {
                ws -= 1;
                if bytes[ws] & 0xC0 != 0x80 {
                    break;
                }
            }
            out.push_str(&input[i..ws]);
            // active window ends at the first not-in-R char (a safe ccc-0/QC=Yes starter) at/after `rel`
            let mut seg_end = rel;
            while seg_end < n {
                let (cp, w) = decode_cp(bytes, seg_end);
                if !bit_set::<R>(cp) {
                    break;
                }
                seg_end += w;
            }
            decompose_to_pairs::<D>(&input[ws..seg_end], buf, pending);
            compose_into(buf, &mut out, marks);
            if seg_end == n {
                break;
            }
            i = seg_end;
            rel = next_set::<R>(bytes, i).0;
            if rel == n {
                out.push_str(&input[i..n]);
                break;
            }
        }
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

/// The dedicated NFD-check classify tag of `cp` — the per-codepoint value baked into
/// `NFD_CHECK_TABLES` (generated by `bitmap_gen`, which calls this; dev-path only, `#[doc(hidden)]`).
/// One tag drives every normalization check with tag arithmetic alone (no decode, no table loads):
///   * `0x00`        — inert: stable under NFD/NFKD, ccc 0, composition-irrelevant.
///   * `0x01..=0x3B` — identity combining mark: the value is its ccc **rank** (order-preserving over
///     the ~55 distinct ccc values, so rank comparisons decide canonical order exactly).
///   * `0x3C`        — compat-changing STARTER (NFKD/NFKC break; NFD/NFC stable), e.g. `ﬁ`.
///   * `0x3D`        — odd mark: compat-changing identity mark — rank via a rare probe under the NFD
///     check; breaks under NFKD; window under compose.
///   * `0x40 | rank` — NFC quick-check **Maybe** (composes as the second of some primary composite):
///     `0x40` alone is a composable starter (V/T jamo); `0x40|r` a composable mark WITH its rank, so
///     decompose order checks stay exact for the common diacriticals.
///   * `0x7E`        — CHANGES under NFD (canonical decomposition, Hangul syllables included).
/// Derived 1:1 from the committed tables, so it can never drift from the normalizer itself.
#[doc(hidden)]
pub fn check_tag(cp: u32) -> u8 {
    let rank = ccc_rank_map();
    // canonical decomposition changes it (Hangul's empty entry included): 0x7E when composition-
    // stable (NFC_QC=Yes — precomposed é/ά/Hangul, the overwhelming majority: compose skips them
    // tag-only), 0x7D when composition-relevant (exclusions/QC≠Yes: compose must open a window).
    if bit_set::<Nfd>(cp) {
        let e = Nfd::entries(cp);
        if e.len() != 1 || (e[0] & 0xFF_FFFF) != cp {
            let ccc0 = (e.first().map(|&x| x >> 24).unwrap_or(0)) as u8;
            let comp_relevant =
                bit_set::<NfcRelevant>(cp) || bit_set::<NfkcRelevant>(cp) || ccc0 != 0;
            return if comp_relevant { 0x7D } else { 0x7E };
        }
    }
    // identity under NFD from here on; ccc rank from the NFD entry (0 if none)
    let r = if bit_set::<Nfd>(cp) { rank[&((Nfd::entries(cp)[0] >> 24) as u8)] } else { 0 };
    // compatibility decomposition changes it
    let nfkd_changes = bit_set::<Nfkd>(cp) && {
        let e = Nfkd::entries(cp);
        e.len() != 1 || (e[0] & 0xFF_FFFF) != cp
    };
    if nfkd_changes {
        return if r == 0 { 0x3C } else { 0x3D }; // compat starter / odd compat mark (rare)
    }
    // NFC quick-check "Maybe": composes as the SECOND char of some primary composite — the baked
    // COMPOSE seconds plus the arithmetic Hangul V/T jamo. Rank rides along in the low bits.
    if maybe_composable(cp) {
        return 0x40 | r;
    }
    r
}

/// ccc → order-preserving rank (1-based; ~55 distinct values). Shared by `check_tag` and the odd-mark
/// probe path. Dev/rare-path only.
#[doc(hidden)]
pub fn ccc_rank_map() -> &'static std::collections::HashMap<u8, u8> {
    use std::sync::OnceLock;
    static RANK: OnceLock<std::collections::HashMap<u8, u8>> = OnceLock::new();
    RANK.get_or_init(|| {
        let mut cccs: Vec<u8> = (0..NFD_CAP)
            .filter(|&c| bit_set::<Nfd>(c))
            .flat_map(|c| Nfd::entries(c).iter().map(|&e| (e >> 24) as u8))
            .filter(|&c| c != 0)
            .collect();
        cccs.sort_unstable();
        cccs.dedup();
        assert!(cccs.len() <= 0x3B, "ccc rank overflow: {} distinct values", cccs.len());
        cccs.iter().enumerate().map(|(i, &c)| (c, i as u8 + 1)).collect()
    })
}

/// NFC quick-check "Maybe": `cp` composes as the SECOND element of some primary composite.
fn maybe_composable(cp: u32) -> bool {
    use std::sync::OnceLock;
    static MAYBE: OnceLock<std::collections::HashSet<u32>> = OnceLock::new();
    MAYBE
        .get_or_init(|| {
            let mut s: std::collections::HashSet<u32> =
                COMPOSE.iter().map(|&(k, _)| (k & 0x1F_FFFF) as u32).collect();
            s.extend(0x1161..=0x1175); // V jamo (L+V composes arithmetically)
            s.extend(0x11A8..=0x11C2); // T jamo (LV+T)
            s
        })
        .contains(&cp)
}

/// Generator for `src/nfd_byte_tables.rs` — the byte-form decomposition tables the SIMD decompose
/// kernel gathers from. Derived 1:1 from the committed `(ccc, char)` trie (never from
/// `unicode-normalization` directly), so the two representations cannot drift. Regenerate with:
///   cargo test -p atomsplit --release gen_byte_tables -- --ignored
///
/// Layout per form: `BYTE_DATA` is slot-parallel to `TRIE_DATA` (same two-level trie indexing);
/// a non-zero slot is `(blob_off << 8) | byte_len` where `BLOB[off..off+len]` is the decomposition
/// as UTF-8 bytes and `BLOB[off-3..off]` is `[first_ccc, last_ccc, mark_run_off]`:
///   * `first_ccc` / `last_ccc` — ccc of the first / last decomposed char (the cross-char canonical-
///     order chain check: fast path requires `first_ccc == 0 || first_ccc >= running last_ccc`).
///   * `mark_run_off` — byte offset just past the LAST starter (0 ⇔ the decomposition is pure marks,
///     in which case the current mark run continues and `run_start` must not move).
/// `byte_len == 0xFF` is the scalar-fallback sentinel (internally unsorted or > 0xFE bytes — none
/// expected; the generator asserts if one appears so we notice). Hangul stays arithmetic: its trie
/// slots are 0 in both tables. The blob ends with 16 zero bytes so unaligned 16-byte loads at any
/// valid `off` never read out of bounds.
#[cfg(test)]
mod gen_byte_tables {
    use super::{Decomp, Nfd, Nfkd};
    use std::collections::HashMap;
    use std::fmt::Write;

    fn gen_form<D: Decomp>(name: &str, o: &mut String) {
        let mut blob: Vec<u8> = Vec::new();
        let mut memo: HashMap<u32, u32> = HashMap::new(); // trie packed → byte packed (dedup: same
        // packed value ⇒ same DECOMP slice ⇒ same bytes, so block sharing in the trie stays coherent)
        let mut data: Vec<u32> = Vec::with_capacity(D::TRIE_DATA.len());
        let mut fallbacks = 0usize;
        for &packed in D::TRIE_DATA {
            if packed == 0 {
                data.push(0);
                continue;
            }
            let v = *memo.entry(packed).or_insert_with(|| {
                let (off, len) = ((packed >> 8) as usize, (packed & 0xFF) as usize);
                let entries = &D::DECOMP[off..off + len];
                let mut bytes = Vec::new();
                let mut mark_off = 0usize; // byte offset just past the last starter
                let (mut prev_ccc, mut sorted) = (0u8, true);
                for &e in entries {
                    let ccc = (e >> 24) as u8;
                    let ch = char::from_u32(e & 0xFF_FFFF).unwrap();
                    if ccc != 0 && ccc < prev_ccc {
                        sorted = false; // would trigger reorder_insert — not blob-copy-safe
                    }
                    let mut buf = [0u8; 4];
                    bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    if ccc == 0 {
                        mark_off = bytes.len();
                    }
                    prev_ccc = ccc; // marks-only compare: a starter resets via ccc==0 < any mark ccc
                }
                let first = (entries[0] >> 24) as u8;
                let last = (entries[entries.len() - 1] >> 24) as u8;
                let blen = if sorted && bytes.len() <= 0xFE { bytes.len() as u32 } else { 0xFF };
                if blen == 0xFF {
                    fallbacks += 1;
                }
                blob.extend_from_slice(&[first, last, mark_off as u8]);
                let boff = blob.len() as u32;
                blob.extend_from_slice(&bytes);
                (boff << 8) | blen
            });
            data.push(v);
        }
        blob.extend_from_slice(&[0u8; 16]); // tail pad: unaligned 16-byte loads never go OOB
        assert_eq!(fallbacks, 0, "{name}: unexpected fallback entries — investigate before shipping");
        writeln!(o, "#[rustfmt::skip]").unwrap();
        write!(o, "pub static {name}_BYTE_DATA: [u32; {}] = [", data.len()).unwrap();
        for (k, v) in data.iter().enumerate() {
            if k > 0 {
                o.push(',');
            }
            write!(o, "{v}").unwrap();
        }
        writeln!(o, "];").unwrap();
        writeln!(o, "#[rustfmt::skip]").unwrap();
        write!(o, "pub static {name}_BYTE_BLOB: [u8; {}] = [", blob.len()).unwrap();
        for (k, v) in blob.iter().enumerate() {
            if k > 0 {
                o.push(',');
            }
            write!(o, "{v}").unwrap();
        }
        writeln!(o, "];").unwrap();
    }

    /// Composed (NFKC) byte blobs for compat-decomposing STARTERS: slot-parallel to the NFKD trie,
    /// `BLOB[off..off+len]` = the char's full NFKC as UTF-8. A slot is baked only when emitting the
    /// blob verbatim can NEVER interact with neighbours: every result char has ccc 0, the first result
    /// char is not QC-Maybe (no back-composition), and the last is not a composition FIRST (no forward
    /// composition with a following Maybe char). Runtime additionally gates on the `0x3C` check tag.
    fn gen_composed(name: &str, o: &mut String) {
        use super::{Decomp, Nfkd};
        use crate::compose_tables::COMPOSE;
        use crate::nfd::{maybe_composable, nfkc};
        let firsts: std::collections::HashSet<u32> = COMPOSE
            .iter()
            .map(|&(k, _)| (k >> 21) as u32)
            .chain(0x1100..=0x1112) // L jamo (L+V composes)
            .chain((0xAC00..=0xD7A3).step_by(28)) // LV syllables (LV+T)
            .collect();
        let mut blob: Vec<u8> = Vec::new();
        let mut memo: HashMap<u32, u32> = HashMap::new();
        let mut data: Vec<u32> = Vec::with_capacity(Nfkd::TRIE_DATA.len());
        for &packed in Nfkd::TRIE_DATA {
            if packed == 0 {
                data.push(0);
                continue;
            }
            let v = *memo.entry(packed).or_insert_with(|| {
                let (off, len) = ((packed >> 8) as usize, (packed & 0xFF) as usize);
                let entries = &Nfkd::DECOMP[off..off + len];
                let s: String = entries
                    .iter()
                    .map(|&e| char::from_u32(e & 0xFF_FFFF).unwrap())
                    .collect();
                let k = nfkc(&s).into_owned(); // nfkc of the NFKD expansion == nfkc of the char
                let chars: Vec<char> = k.chars().collect();
                let ccc_of = |c: char| {
                    let e = Nfkd::entries(c as u32);
                    if e.is_empty() || (e.len() == 1 && (e[0] & 0xFF_FFFF) == c as u32) {
                        e.first().map(|&x| (x >> 24) as u8).unwrap_or(0)
                    } else {
                        1 // decomposes further?! never for an NFKC result — treat as unsafe
                    }
                };
                let safe = !k.is_empty()
                    && k.len() <= 16
                    && chars.iter().all(|&c| ccc_of(c) == 0)
                    && !maybe_composable(chars[0] as u32)
                    && !firsts.contains(&(*chars.last().unwrap() as u32));
                if !safe {
                    return 0;
                }
                let boff = blob.len() as u32;
                blob.extend_from_slice(k.as_bytes());
                (boff << 8) | k.len() as u32
            });
            data.push(v);
        }
        blob.extend_from_slice(&[0u8; 16]);
        writeln!(o, "#[rustfmt::skip]").unwrap();
        write!(o, "pub static {name}_BYTE_DATA: [u32; {}] = [", data.len()).unwrap();
        for (k, v) in data.iter().enumerate() {
            if k > 0 {
                o.push(',');
            }
            write!(o, "{v}").unwrap();
        }
        writeln!(o, "];").unwrap();
        writeln!(o, "#[rustfmt::skip]").unwrap();
        write!(o, "pub static {name}_BYTE_BLOB: [u8; {}] = [", blob.len()).unwrap();
        for (k, v) in blob.iter().enumerate() {
            if k > 0 {
                o.push(',');
            }
            write!(o, "{v}").unwrap();
        }
        writeln!(o, "];").unwrap();
    }

    #[test]
    #[ignore = "writes src/nfd_byte_tables.rs — run explicitly to regenerate"]
    fn generate() {
        let mut o = String::new();
        writeln!(
            o,
            "//! GENERATED — do NOT edit. Byte-form decomposition tables for the SIMD decompose\n\
             //! kernel, derived 1:1 from the committed `(ccc, char)` tries in `nfd_tables.rs` /\n\
             //! `nfkd_tables.rs` (see `nfd.rs::gen_byte_tables` for the layout). Regenerate with:\n\
             //!   cargo test -p atomsplit --release gen_byte_tables -- --ignored\n"
        )
        .unwrap();
        gen_form::<Nfd>("NFD", &mut o);
        gen_form::<Nfkd>("NFKD", &mut o);
        gen_composed("NFKC", &mut o);
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/nfd_byte_tables.rs");
        std::fs::write(path, o).unwrap();
        eprintln!("wrote {path}");
    }
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

    /// The generated byte-form tables round-trip the `(ccc, char)` trie exactly: for every unstable
    /// codepoint under both forms, the blob's UTF-8 decodes to the same char sequence, the header cccs
    /// match the entries' first/last ccc, and `mark_run_off` equals the offset just past the last starter.
    #[test]
    fn byte_tables_round_trip() {
        use super::{Decomp, bit_set};
        fn check<D: Decomp>(form: &str) {
            for cp in 0..D::CAP {
                if !bit_set::<D>(cp) {
                    continue;
                }
                let entries = D::entries(cp);
                let (off, len) = D::byte_entry(cp);
                if entries.is_empty() {
                    assert_eq!((off, len), (0, 0), "{form} U+{cp:04X}: Hangul/empty must have no blob");
                    continue;
                }
                assert_ne!(len, 0xFF, "{form} U+{cp:04X}: unexpected fallback sentinel");
                let bytes = &D::BYTE_BLOB[off..off + len];
                let decoded: Vec<char> = std::str::from_utf8(bytes).unwrap().chars().collect();
                let expect: Vec<char> = entries
                    .iter()
                    .map(|&e| char::from_u32(e & 0xFF_FFFF).unwrap())
                    .collect();
                assert_eq!(decoded, expect, "{form} U+{cp:04X}: blob bytes");
                let (first, last, mark_off) = (
                    D::BYTE_BLOB[off - 3],
                    D::BYTE_BLOB[off - 2],
                    D::BYTE_BLOB[off - 1] as usize,
                );
                assert_eq!(first, (entries[0] >> 24) as u8, "{form} U+{cp:04X}: first_ccc");
                assert_eq!(last, (entries[entries.len() - 1] >> 24) as u8, "{form} U+{cp:04X}: last_ccc");
                let mut expect_off = 0usize;
                let mut w = 0usize;
                for &e in entries {
                    let ch = char::from_u32(e & 0xFF_FFFF).unwrap();
                    w += ch.len_utf8();
                    if (e >> 24) == 0 {
                        expect_off = w;
                    }
                }
                assert_eq!(mark_off, expect_off, "{form} U+{cp:04X}: mark_run_off");
            }
        }
        check::<super::Nfd>("NFD");
        check::<super::Nfkd>("NFKD");
    }

    /// The committed check tables must match `check_tag` exactly (catches a stale
    /// `nfd_check_tables.rs` after a table change), and `check_tag`'s semantics must match the
    /// normalizer's own tables: break-tags ⇔ decomposition changes the char; rank order ⇔ ccc order.
    #[test]
    fn check_tables_exact() {
        use super::{Decomp, bit_set, check_tag};
        let mut buf = [0u8; 4];
        let mut tags = [0u8; 4];
        // rank order preserves ccc order: collect (ccc, rank) pairs and verify monotonicity
        let mut pairs: Vec<(u8, u8)> = Vec::new();
        for cp in 0..0x110000u32 {
            let Some(c) = char::from_u32(cp) else { continue };
            let expect = check_tag(cp);
            let s = c.encode_utf8(&mut buf);
            crate::classify::classify_scalar_with::<0x81>(
                s.as_bytes(),
                &mut tags[..s.len()],
                &crate::nfd_check_tables::NFD_CHECK_TABLES,
            );
            assert_eq!(tags[0], expect, "U+{cp:04X}: committed check table drifted from check_tag");
            // semantics vs the normalizer's tables
            let nfd_changes = bit_set::<super::Nfd>(cp) && {
                let e = super::Nfd::entries(cp);
                e.len() != 1 || (e[0] & 0xFF_FFFF) != cp
            };
            assert_eq!(expect >= 0x7D, nfd_changes, "U+{cp:04X}: NFD-break tag (t = {expect:#04x})");
            let nfkd_changes = bit_set::<super::Nfkd>(cp) && {
                let e = super::Nfkd::entries(cp);
                e.len() != 1 || (e[0] & 0xFF_FFFF) != cp
            };
            assert_eq!(
                super::Nfkd::tag_breaks(expect),
                nfkd_changes,
                "U+{cp:04X}: NFKD-break tag (t = {expect:#04x})"
            );
            // Maybe flag ⇔ composable-second, on non-breaking chars only
            if !nfd_changes && !nfkd_changes {
                assert_eq!(
                    expect & 0x40 != 0,
                    super::maybe_composable(cp),
                    "U+{cp:04X}: Maybe flag (t = {expect:#04x})"
                );
            }
            let r = expect & 0x3F;
            if !super::Nfkd::tag_breaks(expect) && r != 0 {
                let ccc = (super::Nfd::entries(cp)[0] >> 24) as u8;
                assert_ne!(ccc, 0, "U+{cp:04X}: rank tag on a ccc-0 char (t = {expect:#04x})");
                pairs.push((ccc, r));
            }
        }
        pairs.sort_unstable();
        for w in pairs.windows(2) {
            assert!(
                (w[0].0 == w[1].0) == (w[0].1 == w[1].1) && w[0].1 <= w[1].1,
                "rank order must mirror ccc order: {w:?}"
            );
        }
    }

    /// The aarch64 tag-driven kernels are gated behind input-size/content dispatch in `nfd()`/`nfc()`,
    /// so exercise them DIRECTLY: byte-exact vs `unicode-normalization` for every codepoint glued to
    /// mark suffixes (both orders), for all four forms.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn tagged_paths_exhaustive() {
        use super::{Nfd, Nfkd, NfcRelevant, NfkcRelevant, compose_tagged, decompose_tagged};
        let mut buf = String::new();
        for cp in 0u32..0x30000 {
            let Some(c) = char::from_u32(cp) else { continue };
            for suffix in ["", "\u{0301}", "\u{0323}\u{0301}", "\u{0301}\u{0323}", "\u{05B4}"] {
                buf.clear();
                buf.push(c);
                buf.push_str(suffix);
                assert_eq!(
                    decompose_tagged::<Nfd>(&buf),
                    buf.nfd().collect::<String>(),
                    "tagged NFD {cp:#x} {suffix:?}"
                );
                assert_eq!(
                    decompose_tagged::<Nfkd>(&buf),
                    buf.nfkd().collect::<String>(),
                    "tagged NFKD {cp:#x} {suffix:?}"
                );
                assert_eq!(
                    compose_tagged::<Nfd, NfcRelevant>(&buf),
                    buf.nfc().collect::<String>(),
                    "tagged NFC {cp:#x} {suffix:?}"
                );
                assert_eq!(
                    compose_tagged::<Nfkd, NfkcRelevant>(&buf),
                    buf.nfkc().collect::<String>(),
                    "tagged NFKC {cp:#x} {suffix:?}"
                );
            }
        }
    }

    #[ignore = "manual timing probe"]
    fn timing_probe() {
        for (label, rel) in [("Thai", "benches/data/th.txt"), ("Korean", "benches/data/ko.txt")] {
            let path = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), rel);
            let Ok(s) = std::fs::read_to_string(&path) else { continue };
            let mut c = s.len().min(180_000);
            while c > 0 && !s.is_char_boundary(c) {
                c -= 1;
            }
            let text = &s[..c];
            let n = text.len();
            let mut tags = vec![0u8; n];
            let mut best = f64::INFINITY;
            for _ in 0..7 {
                let t = std::time::Instant::now();
                for _ in 0..20 {
                    crate::classify::classify_with::<0x81, 0x80, { crate::classify::NO_CJK }>(
                        text.as_bytes(),
                        &mut tags,
                        &crate::nfd_check_tables::NFD_CHECK_TABLES,
                    );
                    std::hint::black_box(tags[0]);
                }
                best = best.min(t.elapsed().as_nanos() as f64 / (20 * n) as f64);
            }
            eprintln!("{label}: check-classify {best:.3} ns/B");
        }
    }

    /// The streaming3 check only engages for inputs ≥ 256 bytes of 3-byte-lead text, so exercise it
    /// directly with long synthetic paragraphs — normal, mark-clustered, out-of-order, decomposing,
    /// block-hopping — byte-exact vs `unicode-normalization` for NFD/NFKD.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn streaming3_long_inputs() {
        let cases: Vec<String> = vec![
            "\u{0E2A}\u{0E27}\u{0E31}\u{0E2A}\u{0E14}\u{0E35}".repeat(60), // Thai w/ mai han-akat
            "\u{0E01}\u{0E38}\u{0E49}".repeat(80),                    // Thai below-vowel + tone (ccc 103,107)
            "\u{0E01}\u{0E49}\u{0E38}".repeat(80),                    // out of canonical order (107 then 103)
            "\u{0915}\u{094D}\u{0937}\u{093F}".repeat(70),           // Devanagari conjunct + matra
            format!("{}ï{}", "\u{0E44}\u{0E17}\u{0E22}".repeat(40), "\u{0E44}\u{0E17}\u{0E22}".repeat(40)), // stray decomposing latin
            format!("{}{}", "\u{0E01}".repeat(100), "\u{0995}\u{09CD}".repeat(60)), // block hop Thai→Bengali
            "\u{0E33}".repeat(90),                                    // SARA AM (decomposes under NFD? no — but NFKD?)
            format!("abc {} def", "\u{0E01}\u{0E48}".repeat(70)),      // ascii mixing
        ];
        for (k, s) in cases.iter().enumerate() {
            assert!(s.len() >= 256, "case {k} too short");
            assert_eq!(nfd(s.as_str()), s.nfd().collect::<String>(), "streaming3 NFD case {k}");
            assert_eq!(nfkd(s.as_str()), s.nfkd().collect::<String>(), "streaming3 NFKD case {k}");
        }
    }

    #[ignore = "debug probe"]
    fn compose_check_probe() {
        let path = format!("{}/benches/data/th.txt", env!("CARGO_MANIFEST_DIR"));
        let s = std::fs::read_to_string(&path).unwrap();
        let mut c = s.len().min(180_000);
        while c > 0 && !s.is_char_boundary(c) {
            c -= 1;
        }
        let text = &s[..c];
        let ok = super::streaming3_compose_ok::<super::Nfd>(text.as_bytes());
        eprintln!("streaming3_compose_ok(th) = {ok}");
        // find compose-relevant tags scalar
        let t = &crate::nfd_check_tables::NFD_CHECK_TABLES;
        let bytes = text.as_bytes();
        let mut i = 0;
        let mut found = 0;
        while i < bytes.len() && found < 5 {
            let b = bytes[i];
            if b < 0x80 { i += 1; continue; }
            let tag = t.classify_char(bytes, i);
            if tag & 0x40 != 0 || tag == 0x7D || tag == 0x3D {
                let (cp, _) = super::decode_cp(bytes, i);
                eprintln!("relevant at byte {i}: U+{cp:04X} tag={tag:#04x}");
                found += 1;
            }
            i += if b < 0xE0 { 2 } else if b < 0xF0 { 3 } else { 4 };
        }
        eprintln!("(scan done, {found} relevant chars shown)");
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
