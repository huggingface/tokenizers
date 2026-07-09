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
    SymOther = 10,     // non-ASCII \p{S}
    NumericOther = 11, // is_numeric ∖ \p{N}
    Control = 12,      // control ∪ unassigned (split out of SymOther so \p{P}∪\p{S} excludes it)
    Sentinel = 13,     // this lead / block needs to go 1 level deeper
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
        | Control.bit()
        | NumericOther.bit();
    /// `\p{P} ∪ \p{S}` (deepseek) — punctuation + symbol, EXCLUDING control/unassigned.
    pub const PUNCT_SYM: u16 = Connector.bit() | Punct.bit() | Apostrophe.bit() | SymOther.bit();
    /// `\p{L} ∪ \p{M}` (deepseek letter/mark run).
    pub const LETTER_MARK: u16 = Letter.bit() | Mark.bit();
}

/// A per-char tag alphabet + its classify paths. `Atoms` (12-way) is the instance today; another
/// pretokenizer-class scheme is just another `impl`.
pub trait TagScheme {
    /// Number of distinct tags (excluding `CONT`). Atoms ≤ 16 → the FSM remap fits one `vqtbl1`.
    const N_TAGS: usize;
    /// Continuation-byte sentinel for this scheme — written to every non-lead byte, transparent to
    /// this scheme's FSMs. (For `Atoms` this is `Atom::Cont`.)
    const CONT: u8;
    /// Sentinel the SIMD path writes for a multibyte lane it couldn't resolve. This would be
    /// astral emojis etc.
    const MB: u8;
    /// If the whole CJK block (E3–ED) collapses to ONE tag under this scheme, name it here to enable
    /// the fast lead-byte range shortcut (Atoms → `Some(Letter)`). `None` → CJK is resolved by the
    /// per-`(lead,b2-pair)` tables instead (for a scheme that maps CJK blocks to distinct tags).
    const CJK_RANGE_TAG: Option<u8>;

    /// Scalar per-char classifier: tag of the char starting at `text[i]`. ASCII → Look Up Table; 2/3-byte →
    /// bitmap-hit helpers / range search. Width is the engine's job (`char_len`). The single source of
    /// truth: every SIMD table is built from this, and it's the tail/astral fallback.
    fn classify_char(text: &[u8], i: usize) -> u8;

    /// This scheme's classify tables — a `const` set baked at compile time by `build.rs` (the
    /// `bitmap_gen` generator); read by both the SIMD kernel and the scalar reader, never built at runtime.
    fn tables() -> &'static crate::simd_classify::Tables;
}

/// Classify `text` under scheme `S` into `tags` (`tags.len() == text.len()`) — the single arch
/// dispatcher. aarch64: NEON at compile time (baseline). x86_64: AVX-512 VBMI → SSE4.1 → scalar,
/// runtime-detected. wasm32 with `simd128`: SIMD128. Everything else: the portable scalar walk. All
/// paths produce the identical stream.
#[inline]
pub fn classify<S: TagScheme>(text: &[u8], tags: &mut [u8]) {
    debug_assert_eq!(text.len(), tags.len());
    #[cfg(target_arch = "aarch64")]
    unsafe {
        crate::simd_classify::classify_neon::<S>(text, tags)
    }
    #[cfg(target_arch = "x86_64")]
    crate::simd_avx_classify::dispatch::<S>(text, tags);
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        crate::simd_wasm_classify::classify_wasm::<S>(text, tags)
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    classify_scalar::<S>(text, tags);
}

/// The shared scalar walk loop — generic over the scheme. One forward pass; per char-start calls
/// `S::classify_char`, fills continuation bytes with `S::CONT`. Monomorphized per scheme.
pub fn classify_scalar<S: TagScheme>(text: &[u8], tags: &mut [u8]) {
    let n = text.len();
    let mut i = 0;
    while i < n {
        let b = text[i];
        // 0xC0 is 0b1100 0000, the continuation header byte.
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
    const N_TAGS: usize = 13;
    const CONT: u8 = Atom::Cont as u8;
    const MB: u8 = Atom::MultiByte as u8;
    const CJK_RANGE_TAG: Option<u8> = Some(Atom::Letter as u8); // all of CJK → Letter → range shortcut

    /// ┌──────────────────────── OWNER: scalar path ────────────────────────┐
    /// ASCII → LUT; 2/3-byte → bitmap-hit helpers (the matchers: `letter2_hit`/`number2_hit` +
    /// `mark2_hit`/`punct2_hit`/… to add); 3-byte-non-CJK residue → range search.
    #[inline]
    fn classify_char(text: &[u8], i: usize) -> u8 {
        crate::atom_tables::ATOM_TABLES.classify_char(text, i)
    }

    fn tables() -> &'static crate::simd_classify::Tables {
        &crate::atom_tables::ATOM_TABLES
    }
}

#[cfg(test)]
mod classify_tests {
    use super::*;

    /// The byte-exactness gate (spec §8): the SIMD path (NEON on aarch64) MUST equal the scalar walk.
    /// This is the first time the SIMD kernel is runnable — `classify_char` used to be `todo!()`.
    #[test]
    fn simd_matches_scalar_atoms() {
        let unit = "Hello, 世界! ½ + ٠١ Ⅷ café\tнаука ไทย 😀\u{0301}mark _u 'q' ©s ½²¼ 안녕 ";
        let corpus = unit.repeat(40); // >32 B so the SIMD chunk loop actually runs
        let text = corpus.as_bytes();
        let mut simd = vec![0u8; text.len()];
        let mut scalar = vec![0u8; text.len()];
        classify::<Atoms>(text, &mut simd);
        classify_scalar::<Atoms>(text, &mut scalar);
        assert_eq!(simd, scalar, "SIMD classify must be byte-exact vs scalar");
    }

}
