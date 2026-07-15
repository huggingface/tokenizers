use crate::atom_tables::ATOM_TABLES;

/// The per-codepoint "atom" categories or "tags" that are used by the finite state machine to emit
/// spit boundaries.
///
/// Proper tags are obtained by composing `(refine << 4) | coarse`: the **low nibble** is this `Atom` (the coarse class every FSM
/// shares) and the **high nibble** is an optional *refinement* that sub-splits one coarse class for a
/// pattern that needs finer granularity — e.g. o200k needs case, so `Letter` carries `refine::UPPER`
/// (`\p{Lu}∪\p{Lt}`) / `refine::LOWER` (`\p{Ll}`) / `0` (caseless `\p{Lm}\p{Lo}`). The classifier stays
/// agnostic (it just emits the table byte); a coarse consumer collapses the refinement for free —
/// [`in_mask`] masks it off, and the SIMD class path `& 0x0F`s before its 16-entry LUT — so only the FSM
/// that opted in (o200k, via [`refine`]) ever sees the high nibble. Ceiling: ≤16 coarse, ≤15 refinements
/// per class; a >16-coarse scheme would swap the 16-entry LUT for a 256-entry `tbl256` remap.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Atom {
    Letter = 0,          // \p{L}
    NumWord = 1,         // Nd ∪ Nl   (numeric AND \w)
    NumOther = 2,        // No        (numeric, not \w)
    Newline = 3,         // \r \n
    Space = 4,           // 0x20 only
    WsOther = 5,         // \s ∖ {\r\n, 0x20}
    Mark = 6,            // \p{M} ∪ {ZWJ,ZWNJ} ∪ (Other_Alphabetic ∖ L)
    Connector = 7,       // \p{Pc}
    Punct = 8,           // (\p{P} ∖ Pc) ∪ ASCII-symbols, minus 0x27
    Apostrophe = 9,      // 0x27 only
    SymOther = 10,       // non-ASCII \p{S}
    NumericOther = 11,   // is_numeric ∖ \p{N}
    Control = 12,        // control ∪ unassigned (split out of SymOther so \p{P}∪\p{S} excludes it)
    Sentinel = 13,       // this lead / block needs to go 1 level deeper
    MultiByte = 14,      // simd could not resolve this multibyte, use the lookup table
    Cont = 15,           // UTF-8 continuation byte — transparent to every FSM
    AlphaSymMark = 0x16, // 0x16 is low nibble (6) the Mark class, high nibble 1 the refined class
    Zwj = 0x26, // 0x26: coarse Mark (low nibble 6), high nibble 2 — ZWJ/ZWNJ (U+200C/200D). Lets the
    // deepseek/o200k FSMs test the tag instead of peeking `text` for `\p{Cf}` joiners.
    // If we need a new class it will usually fall under the 12 unicode ones. If not we can just define
    // new ones with low and high nibble that are not correlated. You should prioritize using the
    // low nibble 13, 14, 15 just because you would be taking a subclass otherwise.
    // Case refinement of `Letter` (low nibble 0), matching `bitmap_gen`: `\p{Lu}∪\p{Lt}` → 0x10,
    // `\p{Ll}` → 0x20, caseless `\p{Lm}\p{Lo}` → 0x00.
    UpperLetter = 0x10,
    LowerLetter = 0x20,
}

/// High-nibble bit (orthogonal to the case/ASM/ZWJ refinements, all ≤ 0x2x) marking a codepoint in the
/// deepseek Split-2 CJK range (Han U+4E00..9FA5 ∪ Kana U+3040..30FF). `fsm_deepseek` tests it instead of
/// peeking `text`; coarse consumers mask it off with `& 0x0F`.
pub const CJK_BIT: u8 = 0x40;

impl Atom {
    /// Since we only have 16 classes for now, this is a fairly cheap way to get a bitmask over the
    /// class.
    /// This atom's bit in a `u16` class mask.
    #[inline]
    pub const fn bit(self) -> u16 {
        1u16 << (self as u16)
    }
}

/// `true` iff the tag's **coarse** class (low nibble) is in `mask`. Masking `tag & 0x0F` makes every
/// coarse consumer refinement-agnostic (a refined `Letter` like `0x20` still tests as `LETTER`) — and is
/// mandatory, not just a view: a refined tag's raw value can exceed 15, and `1u16 << 0x20` would overflow.
#[inline]
pub const fn in_mask(tag: u8, mask: u16) -> bool {
    mask & (1u16 << (tag & 0x0F)) != 0
}

/// UTF-8 char length from the lead byte. Width is a pure function of the lead — no classification.
#[inline(always)]
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

/// Class masks: a unions of atoms, they are used to combine atoms into what the usual regex
/// usually check.
pub mod mask {
    use super::Atom::*;
    /// `\w` used in Whitespace pretokenizer, a "Word" is: letter | Nd∪Nl | mark | connector.
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
        | Control.bit()
        | NumericOther.bit();
    /// `\p{P} ∪ \p{S}` (deepseek) — punctuation + symbol, EXCLUDING control/unassigned.
    pub const PUNCT_SYM: u16 = Connector.bit() | Punct.bit() | Apostrophe.bit() | SymOther.bit();
    /// `\p{L} ∪ \p{M}` (deepseek letter/mark run).
    pub const LETTER_MARK: u16 = Letter.bit() | Mark.bit();
}

pub const CONT: u8 = Atom::Cont as u8;
pub const MB: u8 = Atom::MultiByte as u8;

/// Classify `text` under scheme `S` into `tags` (`tags.len() == text.len()`) — the single arch
/// dispatcher. aarch64: NEON at compile time (baseline). x86_64: AVX-512 VBMI → SSE4.1 → scalar,
/// runtime-detected. wasm32 with `simd128`: SIMD128. Everything else: the portable scalar walk. All
/// paths produce the identical stream.
#[inline]
pub fn classify(text: &[u8], tags: &mut [u8]) {
    // Hard assert (not debug): the SIMD kernels do raw 16-byte stores into `tags` for full chunks, so a
    // short `tags` is out-of-bounds UB in release — reject it here. This is the sole entry point, so the
    // check guards every arch path below.
    assert!(
        tags.len() >= text.len(),
        "atomsplit::classify: `tags` shorter than `text`"
    );
    #[cfg(target_arch = "aarch64")]
    // SAFETY: `tags.len() >= text.len()` (asserted above); NEON vld1q/vst1q are alignment-free.
    unsafe {
        crate::simd_classify::classify_neon(text, tags)
    }
    #[cfg(target_arch = "x86_64")]
    crate::simd_avx_classify::dispatch(text, tags);
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    // SAFETY: `tags.len() >= text.len()` (asserted above); wasm v128 load/store are alignment-free.
    unsafe {
        crate::simd_wasm_classify::classify_wasm(text, tags)
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    classify_scalar(text, tags);
}

/// The shared scalar walk loop — generic over the scheme. One forward pass; per char-start calls
/// `S::classify_char`, fills continuation bytes with `S::CONT`. The non-SIMD fallback and the byte-exact
/// test oracle for the SIMD kernels — not part of the supported public surface.
#[doc(hidden)]
pub fn classify_scalar(text: &[u8], tags: &mut [u8]) {
    let n = text.len();
    let mut i = 0;
    while i < n {
        let b = text[i];
        // 0xC0 is 0b1100 0000, the utf8 continuation header byte.
        if b & 0xC0 == 0x80 {
            tags[i] = CONT; // stray continuation byte
            i += 1;
            continue;
        }
        tags[i] = ATOM_TABLES.classify_char(text, i);
        let w = char_len(b);
        let mut j = 1;
        while j < w && i + j < n {
            tags[i + j] = CONT;
            j += 1;
        }
        i += w;
    }
}
