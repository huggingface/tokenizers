//! Tag-classify: map every char to an `Atom` in one pass — the shared substrate for all
//! category-based pre-tokenizers (see `TAG_CLASSIFY_SPEC.md`).
//!
//! Two interchangeable paths, **SIMD (NEON)** and **scalar**, that MUST produce identical `tags`
//! (one byte-exact test gates both). The tables they read are arch-independent `const` data: SIMD
//! gathers them with `vqtbl`, scalar bit-tests them (that bit-test is exactly the existing matchers).
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
// classify_atoms — the ONE hot pass. Two paths, identical output (byte-exact gate).
// =================================================================================================

/// Write the `Atom` (as `u8`) of each byte into `tags` (`tags.len() == text.len()`); continuation
/// bytes get `Atom::Cont`. Dispatches to the SIMD path on aarch64, scalar elsewhere.
#[inline]
pub fn classify_atoms(text: &[u8], tags: &mut [u8]) {
    debug_assert_eq!(text.len(), tags.len());
    #[cfg(target_arch = "aarch64")]
    unsafe {
        classify_atoms_neon(text, tags)
    }
    #[cfg(not(target_arch = "aarch64"))]
    classify_atoms_scalar(text, tags);
}

/// ┌───────────────────────── OWNER: SIMD path ─────────────────────────┐
/// NEON classify: ASCII nibble-shuffle → atom; 2-byte via `vqtbl` bitmap membership
/// (`LETTER2`/`NUMBER2`/`MARK2`/…); CJK by lead-byte range; 3-byte-non-CJK via the branchless SIMD
/// range kernel `Σ (cp ≥ tᵢ)`. Continuation lanes = `Cont`. MUST equal `classify_atoms_scalar`.
#[cfg(target_arch = "aarch64")]
unsafe fn classify_atoms_neon(text: &[u8], tags: &mut [u8]) {
    let _ = (text, tags);
    todo!("SIMD classify — TAG_CLASSIFY_SPEC.md §2/§7")
}

/// ┌──────────────────────── OWNER: scalar path ────────────────────────┐
/// Scalar classify: one forward pass; per char-start calls `atom_at`, fills continuations with `Cont`.
pub fn classify_atoms_scalar(text: &[u8], tags: &mut [u8]) {
    let n = text.len();
    let mut i = 0;
    while i < n {
        let b = text[i];
        if b & 0xC0 == 0x80 {
            tags[i] = Atom::Cont as u8; // stray continuation byte
            i += 1;
            continue;
        }
        tags[i] = atom_at(text, i) as u8;
        let w = char_len(b);
        let mut j = 1;
        while j < w && i + j < n {
            tags[i + j] = Atom::Cont as u8;
            j += 1;
        }
        i += w;
    }
}

/// ┌──────────────────────── OWNER: scalar path ────────────────────────┐
/// Per-char classification. ASCII → LUT; 2/3-byte → bitmap-hit helpers (the matchers:
/// `letter2_hit`/`number2_hit` + `mark2_hit`/`punct2_hit`/… to add); 3-byte-non-CJK residue → range
/// search. Width is the caller's job (`char_len`), so this returns only the tag.
#[inline]
pub fn atom_at(text: &[u8], i: usize) -> Atom {
    let _ = (text, i);
    todo!("scalar per-char atom — reuse LETTER2/NUMBER2 bit-tests + ASCII LUT + range tables")
}

// TODO (parallel substrate): `classify_scripts(text, &mut scripts)` — same engine, SCRIPT_RANGES
// table + script-id tags, feeding `fsm::fsm_script_run`. See TAG_CLASSIFY_SPEC.md §3.
