//! Tag-classify: map every char to a small tag in one pass — the shared substrate for all
//! category-based pre-tokenizers (see `TAG_CLASSIFY_SPEC.md`).
//!
//! **One generic engine, parameterized by a `TagScheme`** (turbofish → monomorphized): `Atoms` and
//! `Scripts` are two schemes fed to the same `classify::<S>` — same walk loop, different tables.
//! Two interchangeable paths per scheme, **SIMD (NEON)** and **scalar**, that MUST produce identical
//! `tags` (one byte-exact test gates both). Tables are arch-independent `const` data: SIMD gathers
//! them with `vqtbl`, scalar bit-tests them (that bit-test is exactly the existing matchers).
#![allow(dead_code)] // skeleton — the two paths land incrementally

/// The 12-atom alphabet (fits `u4`). Mutually exclusive; continuation bytes carry `Cont`.
/// Exact Unicode definitions are in `TAG_CLASSIFY_SPEC.md` §1 (derived by enumerating all codepoints).
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Atom {
    Letter = 0,        // \p{L}
    NumWord = 1,       // Nd ∪ Nl   (numeric AND \w)
    NumOther = 2,      // No        (numeric, not \w)
    Newline = 3,       // \r \n
    Space = 4,         // 0x20 only
    WsOther = 5,       // \s ∖ {\r\n, 0x20}
    Mark = 6,          // \p{M} ∪ {ZWJ,ZWNJ} ∪ (Other_Alphabetic ∖ L)
    Connector = 7,     // \p{Pc}
    Punct = 8,         // (\p{P} ∖ Pc) ∪ ASCII-symbols, minus 0x27
    Apostrophe = 9,    // 0x27 only
    SymOther = 10,     // non-ASCII \p{S} ∪ control ∪ unassigned
    NumericOther = 11, // is_numeric ∖ \p{N}
    Sentinel =13,      // this lead / block needs to go 1 level deeper
    MultiByte = 14,    // simd could not resolve this multibyte, use the lookup table
    Cont = 15,         // UTF-8 continuation byte — transparent to every FSM
}

impl Atom {
    /// This atom's bit in a `u16` class mask.
    #[inline]
    pub const fn bit(self) -> u16 {
        1u16 << (self as u16)
    }
}

/// `true` iff the raw tag (`Atom as u8`) is in `mask`.
#[inline]
pub const fn in_mask(tag: u8, mask: u16) -> bool {
    mask & (1u16 << tag) != 0
}

/// UTF-8 char length from the lead byte. Width is a pure function of the lead — no classification.
#[inline]
pub const fn char_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b < 0xE0 {
        2
    } else if b < 0xF0 {
        3
    } else {
        4
    }
}

/// Class masks — unions of atoms, the predicates the FSMs test. `const`, so callers inline them and
/// they double as the const-generic `DELIM`/`DROP` parameters of the FSM shapes.
pub mod mask {
    use super::Atom::*;
    /// `\w` (Whitespace pretokenizer "Word"): letter | Nd∪Nl | mark | connector.
    pub const WORD: u16 = Letter.bit() | NumWord.bit() | Mark.bit() | Connector.bit();
    /// `\s`: newline + 0x20 + other whitespace.
    pub const WS: u16 = Newline.bit() | Space.bit() | WsOther.bit();
    /// `[\r\n]` only.
    pub const NEWLINE: u16 = Newline.bit();
    /// `\p{L}`.
    pub const LETTER: u16 = Letter.bit();
    /// `\p{N}`.
    pub const NUMBER: u16 = NumWord.bit() | NumOther.bit();
    /// `is_numeric` (Digits pretokenizer).
    pub const NUMERIC: u16 = NumWord.bit() | NumOther.bit() | NumericOther.bit();
    /// `is_ascii_punctuation | is_punctuation` (Punctuation / Bert).
    pub const PUNCT: u16 = Connector.bit() | Punct.bit() | Apostrophe.bit();
    /// cl100k `[^\s\p{L}\p{N}]` — the rule-4 punct-run class.
    pub const NOT_WS_L_N: u16 = Mark.bit()
        | Connector.bit()
        | Punct.bit()
        | Apostrophe.bit()
        | SymOther.bit()
        | NumericOther.bit();
}

// =================================================================================================
// The generic engine: classify::<S: TagScheme>.  One walk loop, one hot pass; the scheme supplies
// the tables (as its scalar `classify_char` + its SIMD `classify_neon`). Turbofish monomorphizes it.
// =================================================================================================

/// A per-char tag alphabet + its two classify paths. `Atoms` (12-way) and `Scripts` (script-id) are
/// the two instances; a third pretokenizer-class scheme is just another `impl`.
pub trait TagScheme {
    /// Number of distinct tags (excluding `CONT`). Atoms ≤ 16 → the FSM remap fits one `vqtbl1`.
    const N_TAGS: usize;
    /// Continuation-byte sentinel for this scheme — written to every non-lead byte, transparent to
    /// this scheme's FSMs. (For `Atoms` this is `Atom::Cont`.)
    const CONT: u8;

    /// Scalar per-char classifier: tag of the char starting at `text[i]`. ASCII → LUT; 2/3-byte →
    /// bitmap-hit helpers / range search. Width is the engine's job (`char_len`).
    fn classify_char(text: &[u8], i: usize) -> u8;

    /// SIMD (NEON) whole-buffer classify for this scheme. Writes `tags[..text.len()]`; continuation
    /// lanes = `CONT`. MUST equal the scalar walk over `classify_char`.
    #[cfg(target_arch = "aarch64")]
    unsafe fn classify_neon(text: &[u8], tags: &mut [u8]);
}

/// Classify `text` under scheme `S` into `tags` (`tags.len() == text.len()`). SIMD on aarch64,
/// scalar elsewhere — both paths produce the identical stream. `classify::<Atoms>(t, &mut tags)`.
#[inline]
pub fn classify<S: TagScheme>(text: &[u8], tags: &mut [u8]) {
    debug_assert_eq!(text.len(), tags.len());
    #[cfg(target_arch = "aarch64")]
    unsafe {
        S::classify_neon(text, tags)
    }
    #[cfg(not(target_arch = "aarch64"))]
    classify_scalar::<S>(text, tags);
}

/// The shared scalar walk loop — generic over the scheme. One forward pass; per char-start calls
/// `S::classify_char`, fills continuation bytes with `S::CONT`. Monomorphized per scheme.
pub fn classify_scalar<S: TagScheme>(text: &[u8], tags: &mut [u8]) {
    let n = text.len();
    let mut i = 0;
    while i < n {
        let b = text[i];
        if b & 0xC0 == 0x80 {
            tags[i] = S::CONT; // stray continuation byte
            i += 1;
            continue;
        }
        tags[i] = S::classify_char(text, i);
        let w = char_len(b);
        let mut j = 1;
        while j < w && i + j < n {
            tags[i + j] = S::CONT;
            j += 1;
        }
        i += w;
    }
}

// ── Scheme: Atoms (the 12-way category alphabet) ───────────────────────────────────────────────

/// The atom scheme fed to `classify::<Atoms>`. Zero-sized; carries the tables via its impl.
pub struct Atoms;

impl TagScheme for Atoms {
    const N_TAGS: usize = 12;
    const CONT: u8 = Atom::Cont as u8;

    /// ┌──────────────────────── OWNER: scalar path ────────────────────────┐
    /// ASCII → LUT; 2/3-byte → bitmap-hit helpers (the matchers: `letter2_hit`/`number2_hit` +
    /// `mark2_hit`/`punct2_hit`/… to add); 3-byte-non-CJK residue → range search.
    #[inline]
    fn classify_char(text: &[u8], i: usize) -> u8 {
        let _ = (text, i);
        todo!("scalar per-char atom — reuse LETTER2/NUMBER2 bit-tests + ASCII LUT + range tables")
    }

    /// ┌───────────────────────── OWNER: SIMD path ─────────────────────────┐
    /// ASCII nibble-shuffle → atom; 2-byte via `vqtbl` bitmap membership (`LETTER2`/`NUMBER2`/`MARK2`/…);
    /// CJK by lead-byte range; 3-byte-non-CJK via the branchless SIMD range kernel `Σ (cp ≥ tᵢ)`.
    #[cfg(target_arch = "aarch64")]
    unsafe fn classify_neon(text: &[u8], tags: &mut [u8]) {
        use super::simd_classify::classify_neon;
        unsafe { classify_neon(text, tags) }
    }
}

// ── Scheme: Scripts (the parallel substrate; same engine, SCRIPT_RANGES table) ─────────────────

/// The script scheme fed to `classify::<Scripts>` (UnicodeScripts). Tags are script-ids; the FSM is
/// `fsm::fsm_script_run` (id-equality + transparent set), not the atom masks. See spec §3.
pub struct Scripts;

impl TagScheme for Scripts {
    const N_TAGS: usize = 160; // ~Unicode script count; script-ids don't fit a u16 mask (FSM uses id equality)
    const CONT: u8 = 0xFF; // reserved continuation sentinel (never a valid script id)

    /// ┌──────────────────────── OWNER: scalar path ────────────────────────┐
    /// SIMD range kernel's scalar twin: codepoint → script-id via `SCRIPT_RANGES` (the `get_script`
    /// range match), then `fixed_script` remap (Hira/Kata→Han, space→Any).
    #[inline]
    fn classify_char(text: &[u8], i: usize) -> u8 {
        let _ = (text, i);
        todo!("scalar script-id — SCRIPT_RANGES range search + fixed_script remap")
    }

    /// ┌───────────────────────── OWNER: SIMD path ─────────────────────────┐
    /// Same lane structure as `Atoms` (ASCII/CJK fast lanes reused), but tables → script-ids and the
    /// 2/3-byte/general lanes use the SIMD range kernel over `SCRIPT_RANGES`.
    #[cfg(target_arch = "aarch64")]
    unsafe fn classify_neon(text: &[u8], tags: &mut [u8]) {
        let _ = (text, tags);
        todo!("SIMD script classify — reuse ASCII/CJK lanes; range kernel for the rest; spec §3")
    }
}
