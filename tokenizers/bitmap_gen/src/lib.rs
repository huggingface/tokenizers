//! bitmap_gen — generator for atomsplit's shared classify tables (TAG_CLASSIFY_SPEC.md §1/§7).
//!
//! Dev-time only (depends on `unicode-properties`); nothing here is linked into the runtime crate.
//! `cargo run -p bitmap_gen` calls [`generate_atom_tables`] and writes the committed
//! `atomsplit/src/atom_tables.rs`. It bakes the dense `Tables` layout (ascii / 2-byte group / 3-byte
//! fast3 / bmp_rle / astral) read by BOTH the SIMD kernel (`vqtbl`) and the scalar reader
//! (`Tables::classify_char`); the per-codepoint value is a `u8` tag — low nibble = coarse `Atom`, high
//! nibble = optional refinement (o200k case on `Letter`, `ALPHA_SYM` on `Mark`; `0` = none).
use std::fmt::Write as _;
use unicode_properties::{GeneralCategory, UnicodeGeneralCategory};

// Values returned here correspond to src/classify.rs, straight from TAG_CLASSIFY_SPEC.md §1.
// SINGLE current-Unicode source: general categories from `unicode-properties`; the three derived
// *properties* it doesn't carry (White_Space / Alphabetic / Numeric) come from std, which is also
// current — so there is no stale-version mix (the old `unicode_categories` was frozen at Unicode 9.0,
// which mis-tagged every post-9.0 letter as an `ALPHA_SYM` symbol — see the review).
fn atom(c: char) -> u8 {
    use GeneralCategory::*;
    let cp = c as u32;
    match c.general_category() {
        // \p{L}: low nibble Letter(0); high nibble = o200k case refinement (Lu∪Lt=1, Ll=2, Lm∪Lo=0).
        // Coarse consumers mask it off (`in_mask` / SIMD `& 0x0F`); only `fsm_o200k` reads the nibble.
        UppercaseLetter | TitlecaseLetter => return 0x10,
        LowercaseLetter => return 0x20,
        ModifierLetter | OtherLetter => return 0,
        // \p{N}: Nd∪Nl are numeric AND `\w` (NumWord); No is numeric-only (NumOther).
        DecimalNumber | LetterNumber => return 1,
        OtherNumber => return 2,
        _ => {}
    }
    if cp == 0x0A || cp == 0x0D {
        return 3;
    }
    if cp == 0x20 {
        return 4;
    }
    if c.is_whitespace() {
        return 5; // \s ∖ {\r\n, 0x20}
    }
    if matches!(
        c.general_category(),
        NonspacingMark | SpacingMark | EnclosingMark
    ) || cp == 0x200C
        || cp == 0x200D
    {
        return 6; // \p{M} ∪ {ZWJ,ZWNJ}: real marks — ∈ deepseek/o200k `[\p{L}\p{M}]` and ∈ `\w`
    }
    if c.is_alphabetic() {
        // Other_Alphabetic non-mark (circled letters Ⓘ …, category So): a `\w` word char (coarse Mark →
        // WORD mask sees it) but NOT `[\p{L}\p{M}]`. High nibble = refine::ALPHA_SYM(1) tells deepseek /
        // o200k to treat it as the `\p{S}` symbol it categorically is, not a letter/mark.
        return 6 | (1 << 4); // 0x16: coarse Mark, refine ALPHA_SYM
    }
    if c.general_category() == ConnectorPunctuation {
        return 7; // \p{Pc}
    }
    if cp == 0x27 {
        return 9; // apostrophe
    }
    if matches!(
        c.general_category(),
        DashPunctuation
            | OpenPunctuation
            | ClosePunctuation
            | InitialPunctuation
            | FinalPunctuation
            | OtherPunctuation
    ) {
        return 8; // \p{P} ∖ Pc
    }
    if cp < 0x80 && c.is_ascii_punctuation() {
        return 8; // ASCII symbols ($ + < = > ^ ` | ~) that `[^\s\p{L}\p{N}]` / is_punc treats as punct
    }
    if c.is_numeric() {
        return 11; // NumericOther: is_numeric ∖ \p{N}
    }
    if matches!(
        c.general_category(),
        MathSymbol | CurrencySymbol | ModifierSymbol | OtherSymbol
    ) {
        return 10; // non-ASCII \p{S}
    }
    12 // control ∪ unassigned
}

// codepoint encoded by `bytes` (synthetic UTF-8 built by the table loops). The caller's classifier
// maps it (surrogates handled there via `char::from_u32`).
fn cp_of(bytes: &[u8]) -> u32 {
    match bytes.len() {
        1 => bytes[0] as u32,
        // extract the utf8 headers to reconstruct the u32
        2 => ((bytes[0] as u32 & 0x1F) << 6) | (bytes[1] as u32 & 0x3F),
        _ => {
            ((bytes[0] as u32 & 0x0F) << 12)
                | ((bytes[1] as u32 & 0x3F) << 6)
                | (bytes[2] as u32 & 0x3F)
        }
    }
}

/// Build the dense tables from a per-codepoint `classify`, self-validate the scalar reader reproduces
/// `classify(cp)` for all 1.1M codepoints (panics → fails the build on any mismatch), and return the
/// Rust source for `pub static {struct_name}: Tables` (the committed `atom_tables.rs`). `classify` is the
/// single source of truth; every table (and the SIMD kernel) is derived from it. `kind` labels the
/// emitted doc header.
// fixed-dim UTF-8 table generator: `[8][4][64]` index math reads clearer than iterator chains.
#[allow(clippy::needless_range_loop)]
fn generate_tables(struct_name: &str, kind: &str, classify: &dyn Fn(u32) -> u8) -> String {
    let tag = |bytes: &[u8]| classify(cp_of(bytes));

    // ASCII 128-entry table (two halves for the subtract-trick lookup)
    let mut ascii_lo = [0u8; 64];
    let mut ascii_hi = [0u8; 64];
    for b in 0..64u8 {
        ascii_lo[b as usize] = tag(&[b]);
        ascii_hi[b as usize] = tag(&[64 + b]);
    }

    // 2-byte: 8 groups × 4 leads × 64 conts
    let mut group = [[[0u8; 64]; 4]; 8];
    for g in 0..8usize {
        // 0xC0 is 110..... the utf8 header
        // g is the group: 110xxx.. *4 shifts it by 2
        let base = 0xC0u8 + (g as u8) * 4;
        for k in 0..4usize {
            for c in 0..64u8 {
                // 0x80 is 10yyyyyy, its constructing the 2nd byte.
                // We add `k` as its the sub group (0,1,2,3)
                group[g][k][c as usize] = tag(&[base + k as u8, 0x80 | c]);
            }
        }
    }
    // Finally we build the hardest of them all:
    let mut fast3_uni = [0u8; 512];
    let mut fast3_slot = [0u16; 512];
    let mut fast3_mixed: Vec<([u8; 64], [u8; 64])> = Vec::new();
    for lead in 0xE0u8..=0xEF {
        for p in 0..32u8 {
            let (b2e, b2o) = (0x80 + 2 * p, 0x80 + 2 * p + 1);
            let (mut lo, mut hi) = ([0u8; 64], [0u8; 64]);
            for c in 0..64u8 {
                lo[c as usize] = tag(&[lead, b2e, 0x80 | c]);
                hi[c as usize] = tag(&[lead, b2o, 0x80 | c]);
            }
            let idx = (lead - 0xE0) as usize * 32 + p as usize;
            if lo.iter().chain(hi.iter()).all(|&x| x == lo[0]) {
                fast3_uni[idx] = lo[0];
            } else {
                fast3_uni[idx] = 0xFF;
                fast3_slot[idx] = fast3_mixed.len() as u16;
                fast3_mixed.push((lo, hi));
            }
        }
    }

    // Cold fallback: dense BMP tag → run-length encode (emit only on tag change)
    let mut bmp_rle: Vec<(u16, u8)> = Vec::new();
    for cp in 0..0x10000u32 {
        let a = classify(cp);
        if bmp_rle.last().is_none_or(|&(_, la)| la != a) {
            bmp_rle.push((cp as u16, a));
        }
    }

    // astral (cp >= 0x10000), run-length encoded (start_cp, tag).
    let mut astral: Vec<(u32, u8)> = Vec::new();
    for cp in 0x10000u32..=0x10FFFF {
        let a = classify(cp);
        if astral.last().is_none_or(|&(_, la)| la != a) {
            astral.push((cp, a));
        }
    }

    // ── self-validation: the scalar reader (as `Tables::classify_char` indexes these) == classify(cp) ──
    let read = |cp: u32| -> u8 {
        if cp < 0x80 {
            return if cp < 64 {
                ascii_lo[cp as usize]
            } else {
                ascii_hi[(cp - 64) as usize]
            };
        }
        if cp < 0x800 {
            let b0 = 0xC0 | (cp >> 6) as u8;
            return group[((b0 >> 2) & 7) as usize][(b0 & 3) as usize][(cp & 0x3F) as usize];
        }
        if cp < 0x10000 {
            let b0 = 0xE0 | (cp >> 12) as u8;
            let b1 = ((cp >> 6) & 0x3F) as u8;
            let block = (b0 - 0xE0) as usize * 32 + ((b1 >> 1) & 0x1F) as usize;
            let uni = fast3_uni[block];
            if uni != 0xFF {
                return uni;
            }
            let (lo, hi) = &fast3_mixed[fast3_slot[block] as usize];
            return if b1 & 1 == 0 {
                lo[(cp & 0x3F) as usize]
            } else {
                hi[(cp & 0x3F) as usize]
            };
        }
        astral[astral.partition_point(|&(s, _)| s <= cp) - 1].1
    };
    for cp in 0..=0x10FFFFu32 {
        if (0xD800..=0xDFFF).contains(&cp) {
            continue;
        }
        assert_eq!(
            read(cp),
            classify(cp),
            "bitmap_gen: table mismatch at cp={cp:#06x}"
        );
    }

    // ── emit the Rust source ──
    let mut o = String::new();
    o.push_str(
        "//! GENERATED by `bitmap_gen` — do NOT edit. Regenerate with `cargo run -p bitmap_gen`.\n",
    );
    writeln!(o, "//! Dense {kind} classify tables shared by the SIMD kernel (`vqtbl`) and the scalar reader").unwrap();
    o.push_str("//! `Tables::classify_char`. `bitmap_gen` self-validates all 1.1M codepoints. Spec §1/§7.\n");
    o.push_str("use crate::simd_classify::Tables;\n\n");

    let emit_u8 = |o: &mut String, name: &str, t: &[u8]| {
        write!(o, "#[rustfmt::skip]\nstatic {name}: [u8; {}] = [", t.len()).unwrap();
        for (i, v) in t.iter().enumerate() {
            if i > 0 {
                o.push(',');
            }
            write!(o, "{v}").unwrap();
        }
        o.push_str("];\n");
    };

    o.push_str("#[rustfmt::skip]\nstatic GROUP: [[[u8; 64]; 4]; 8] = [");
    for g in 0..8 {
        o.push('[');
        for k in 0..4 {
            o.push('[');
            for c in 0..64 {
                if c > 0 {
                    o.push(',');
                }
                write!(o, "{}", group[g][k][c]).unwrap();
            }
            o.push(']');
            if k < 3 {
                o.push(',');
            }
        }
        o.push(']');
        if g < 7 {
            o.push(',');
        }
    }
    o.push_str("];\n");

    emit_u8(&mut o, "ASCII_LO", &ascii_lo);
    emit_u8(&mut o, "ASCII_HI", &ascii_hi);
    emit_u8(&mut o, "FAST3_UNI", &fast3_uni);

    o.push_str("#[rustfmt::skip]\nstatic FAST3_SLOT: [u16; 512] = [");
    for (i, v) in fast3_slot.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        write!(o, "{v}").unwrap();
    }
    o.push_str("];\n");

    write!(
        o,
        "#[rustfmt::skip]\nstatic FAST3_MIXED: [([u8; 64], [u8; 64]); {}] = [",
        fast3_mixed.len()
    )
    .unwrap();
    for (i, (lo, hi)) in fast3_mixed.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        o.push('(');
        for (half, arr) in [lo, hi].iter().enumerate() {
            o.push('[');
            for (j, v) in arr.iter().enumerate() {
                if j > 0 {
                    o.push(',');
                }
                write!(o, "{v}").unwrap();
            }
            o.push(']');
            if half == 0 {
                o.push(',');
            }
        }
        o.push(')');
    }
    o.push_str("];\n");

    write!(
        o,
        "#[rustfmt::skip]\nstatic BMP_RLE: [(u16, u8); {}] = [",
        bmp_rle.len()
    )
    .unwrap();
    for (i, (s, a)) in bmp_rle.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        write!(o, "(0x{s:X},{a})").unwrap();
    }
    o.push_str("];\n");

    write!(
        o,
        "#[rustfmt::skip]\nstatic ASTRAL: [(u32, u8); {}] = [",
        astral.len()
    )
    .unwrap();
    for (i, (s, a)) in astral.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        write!(o, "(0x{s:X},{a})").unwrap();
    }
    o.push_str("];\n\n");

    writeln!(o, "pub static {struct_name}: Tables = Tables {{").unwrap();
    o.push_str("    ascii_lo: ASCII_LO,\n    ascii_hi: ASCII_HI,\n    group_tables: GROUP,\n");
    o.push_str("    fast3_uni: FAST3_UNI,\n    fast3_slot: FAST3_SLOT,\n");
    o.push_str(
        "    fast3_mixed: &FAST3_MIXED,\n    bmp_rle: &BMP_RLE,\n    astral: &ASTRAL,\n};\n",
    );
    o
}

/// Atom tables — the 12-way category alphabet (`ATOM_TABLES`), with the `Letter` coarse class carrying an
/// o200k case refinement in its high nibble (see [`atom`]).
pub fn generate_atom_tables() -> String {
    generate_tables("ATOM_TABLES", "atom", &|cp| {
        char::from_u32(cp).map(atom).unwrap_or(10)
    })
}

// ── Normalization classifier ───────────────────────────────────────────────────────────────────
//
// A second `Tables` scheme (`NORM_TABLES`) whose per-codepoint `u8` is a *bitmask* of the properties
// the normalizers branch on. `INERT == 0` means the char is unchanged by every normalization rule, so
// the runtime can `memcpy` whole inert runs and only touch flagged chars. The bits mirror EXACTLY the
// functions the tk-encode normalizers call at runtime — `unicode-normalization` `nfd`/`nfkd`,
// `unicode_categories` `is_mark_nonspacing`/`is_other`, std `to_lowercase`, the bert CJK ranges — so a
// char is inert here IFF the legacy `NormalizedString` path leaves it unchanged. That byte-exact
// coupling is why norm_tag deliberately uses the same (Unicode-9.0) `unicode_categories` the normalizer
// does, rather than the current-Unicode `unicode-properties` used for the atom tables.
//
// KEEP IN SYNC with `atomsplit::norm_classify::bit` (identical values). Max tag 0x7D (< 0xFF, the fast3
// "mixed" sentinel), since NFD/NFKD are mutually exclusive.
pub mod norm_bit {
    /// canonical decomposition changes the char (NFD/NFKD/NFC/NFKC; bert strip_accents runs NFD).
    pub const NFD: u8 = 1 << 0;
    /// NFD-stable but *compatibility* decomposition changes it (adds to NFKD/NFKC over NFD).
    pub const NFKD: u8 = 1 << 1;
    /// combining mark (`is_combining_mark`) — `StripAccents` removes it; bert `strip_accents` refines to
    /// nonspacing marks (Mn) at runtime. Superset of Mn.
    pub const MARK: u8 = 1 << 2;
    /// has a lowercase mapping (`Lowercase` normalizer, bert lowercase).
    pub const LOWER: u8 = 1 << 3;
    /// CJK ideograph — bert `handle_chinese_chars` puts spaces around it.
    pub const CJK: u8 = 1 << 4;
    /// bert `clean_text` removes it entirely (NUL, U+FFFD, control).
    pub const CTRL: u8 = 1 << 5;
    /// whitespace — bert `clean_text` folds it to `' '` (also the `Strip` normalizer).
    pub const WS: u8 = 1 << 6;
}

/// bert `handle_chinese_chars` CJK ranges (mirror of `tk-encode`'s `is_chinese_char`).
fn norm_is_chinese(c: char) -> bool {
    matches!(
        c as u32,
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F | 0x2B920..=0x2CEAF | 0xF900..=0xFAFF | 0x2F800..=0x2FA1F
    )
}

/// The per-codepoint normalization property bitmask. Single source of truth for `NORM_TABLES`.
fn norm_tag(c: char) -> u8 {
    use norm_bit::*;
    use unicode_categories::UnicodeCategories;
    use unicode_normalization::char::canonical_combining_class;
    use unicode_normalization::UnicodeNormalization;
    let mut t = 0u8;
    // NFD changes this char's *content* (it decomposes) OR its *order* (nonzero canonical combining
    // class → NFD reorders it). The order case is essential for memcpy-safety: inert runs are copied
    // verbatim, so a reorderable char (e.g. the 26 `Mc` viramas/Hangul tone marks with ccc≠0 that aren't
    // Mn) left inert could land in the wrong position vs a full NFD. `else` = compatibility-only (NFKD).
    if !c.nfd().eq(std::iter::once(c)) || canonical_combining_class(c) != 0 {
        t |= NFD;
    } else if !c.nfkd().eq(std::iter::once(c)) {
        t |= NFKD;
    }
    // MARK = any combining mark (what `StripAccents` removes). Uses the SAME predicate/crate StripAccents
    // calls so the classifier is byte-exact; it's a superset of Mn, and bert refines back to Mn at runtime.
    if unicode_normalization_alignments::char::is_combining_mark(c) {
        t |= MARK;
    }
    // has a lowercase mapping: to_lowercase(c) is anything other than the single char c.
    let mut low = c.to_lowercase();
    if low.next() != Some(c) || low.next().is_some() {
        t |= LOWER;
    }
    if norm_is_chinese(c) {
        t |= CJK;
    }
    // bert clean_text removal: NUL, U+FFFD, and controls (is_other minus \t\n\r).
    if c == '\0' || c == '\u{FFFD}' || (!matches!(c, '\t' | '\n' | '\r') && c.is_other()) {
        t |= CTRL;
    }
    // bert whitespace that clean_text actually CHANGES: \t\n\r + Unicode White_Space folded to ' ',
    // EXCLUDING ' ' itself (it folds to ' ', unchanged) so a plain space stays inert (borrowable).
    if c != ' ' && (matches!(c, '\t' | '\n' | '\r') || c.is_whitespace()) {
        t |= WS;
    }
    t
}

/// Normalization classifier tables — a per-codepoint property bitmask (`norm_bit`); `0` = inert.
/// Regenerated alongside the atom tables by `cargo run -p bitmap_gen`.
pub fn generate_norm_tables() -> String {
    generate_tables("NORM_TABLES", "normalization", &|cp| {
        char::from_u32(cp).map(norm_tag).unwrap_or(0)
    })
}

// ── Decomposition tables (NFD + NFKD) ────────────────────────────────────────────────────────────
//
// The pure-Rust decompose kernel (`atomsplit::nfd`) does NOT classify. It bulk-skips runs that are
// stable under the target form and only decomposes the rest. "Stable" = does not decompose AND ccc == 0
// — byte-exact with `unicode-normalization`. NFD and NFKD share this exact layout, differing only in the
// decomposition function (canonical `nfd` vs compatibility `nfkd`):
//   {NFD,NFKD}_UNSTABLE  bitset over cp < {…}_CAP; bit set = unstable. Probed per SIMD lane (hot path).
//   {NFD,NFKD}_DECOMP    flattened decompositions; each entry = (ccc << 24) | cp (cp 21 bits, ccc 8).
//   {NFD,NFKD}_INDEX     sorted (cp, (off << 8) | len) into {…}_DECOMP — the cold decompose lookup.
// Hangul syllables are unstable in the bitset but ABSENT from the index: the kernel decomposes them by
// S_BASE arithmetic (their canonical == compatibility decomposition), so baking ~11k entries is waste.

const HANGUL: std::ops::RangeInclusive<u32> = 0xAC00..=0xD7A3;

/// Is `c` unstable under NFD (decomposes canonically or reorders)? Mirrors `norm_tag`'s `NFD` bit.
fn nfd_unstable(c: char) -> bool {
    use unicode_normalization::char::canonical_combining_class;
    use unicode_normalization::UnicodeNormalization;
    !c.nfd().eq(std::iter::once(c)) || canonical_combining_class(c) != 0
}
/// Is `c` unstable under NFKD (decomposes canonically or compatibly, or reorders)?
fn nfkd_unstable(c: char) -> bool {
    use unicode_normalization::char::canonical_combining_class;
    use unicode_normalization::UnicodeNormalization;
    !c.nfkd().eq(std::iter::once(c)) || canonical_combining_class(c) != 0
}

/// Emit a `#[rustfmt::skip] pub static NAME: [TY; N] = [..];` array of `Display` values.
fn emit_slice<T: std::fmt::Display>(o: &mut String, ty: &str, name: &str, t: &[T]) {
    write!(o, "#[rustfmt::skip]\npub static {name}: [{ty}; {}] = [", t.len()).unwrap();
    for (i, v) in t.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        write!(o, "{v}").unwrap();
    }
    o.push_str("];\n");
}

/// Baked "bit set = unstable" bitset over cp `< cap`, sized so the highest unstable cp fits and nothing
/// above `cap` is unstable (asserted). Used for the {NFD,NFKD} tables and the NFC/NFKC relevance bitsets.
fn bitset_up_to(pred: &dyn Fn(char) -> bool) -> (u32, Vec<u64>) {
    let max = (0..0x110000u32)
        .filter(|&cp| char::from_u32(cp).is_some_and(pred))
        .next_back()
        .expect("predicate matched no codepoint");
    let cap = (max / 64 + 1) * 64;
    let mut bits = vec![0u64; (cap / 64) as usize];
    for cp in 0..cap {
        if char::from_u32(cp).is_some_and(pred) {
            bits[(cp / 64) as usize] |= 1 << (cp % 64);
        }
    }
    (cap, bits)
}

/// Emit `{FORM}_CAP/_UNSTABLE/_DECOMP/_TRIE_INDEX/_TRIE_DATA` for decomposition `form` ("NFD"/"NFKD"),
/// using `decompose` (`c.nfd()`/`c.nfkd()`) + `unstable`. The decomposition lookup is a TWO-LEVEL TRIE
/// (`DATA[INDEX[cp >> 6] + (cp & 63)]`, one `u32` = `(off << 8) | len` into `DECOMP`, `0` = absent) — an
/// O(1) gather that replaced a sorted-index binary search (~12 branches/char) and closed the gap to
/// xxUTF on decompose-heavy scripts. Self-validates every codepoint against `decompose`.
fn generate_decomp_tables(
    form: &str,
    decompose: &dyn Fn(char) -> Vec<char>,
    unstable: &dyn Fn(char) -> bool,
) -> String {
    use unicode_normalization::char::canonical_combining_class as ccc;

    let (cap, bitset) = bitset_up_to(unstable);
    // Each decomposition output char is one u32 entry: (ccc << 24) | cp. Compact (a u64 inline-UTF-8
    // variant that would let the runtime `memcpy` instead of re-encoding was tried and REGRESSED the
    // decompose-heavy cluster — the 2× table doubled DECOMP cache traffic and lost more than encode cost).
    let mut decomp: Vec<u32> = Vec::new();
    let mut packed = vec![0u32; cap as usize]; // per-cp (off << 8) | len, 0 = stable/absent
    // Max UTF-8 byte expansion (out_bytes / in_bytes, rounded up) over ALL codepoints. Seeded with 3 for
    // Hangul (3-byte syllable → up to three 3-byte jamo = 9 bytes). Baked so the runtime can reserve a
    // provably realloc-free output buffer of `input_len * MAX_EXPAND`.
    let mut max_expand: u32 = 3;
    for cp in 0..cap {
        let Some(c) = char::from_u32(cp) else { continue };
        if !unstable(c) || HANGUL.contains(&cp) {
            continue; // stable → skip; Hangul → arithmetic at runtime, not baked
        }
        let d = decompose(c);
        assert!(d.len() < 256, "{form} decomp of {cp:#x} too long: {}", d.len());
        let (out_bytes, in_bytes) = (d.iter().map(|c| c.len_utf8()).sum::<usize>(), c.len_utf8());
        max_expand = max_expand.max(out_bytes.div_ceil(in_bytes) as u32);
        let off = decomp.len() as u32;
        for &ch in &d {
            decomp.push((ccc(ch) as u32) << 24 | ch as u32);
        }
        // off << 8 | len is never 0 for a real entry: len >= 1, so the "0 = absent" sentinel is safe.
        packed[cp as usize] = off << 8 | d.len() as u32;
    }

    // Two-level trie: block 0 of DATA is the shared all-absent block; non-empty 64-cp blocks are appended.
    let nblocks = (cap / 64) as usize;
    let mut trie_index = vec![0u32; nblocks];
    let mut trie_data: Vec<u32> = vec![0u32; 64];
    for blk in 0..nblocks {
        let block = &packed[blk * 64..blk * 64 + 64];
        if block.iter().any(|&p| p != 0) {
            trie_index[blk] = trie_data.len() as u32;
            trie_data.extend_from_slice(block);
        } // else stays 0 → shared empty block
    }
    let trie_lookup = |cp: u32| -> Option<&[u32]> {
        if cp >= cap {
            return None;
        }
        let p = trie_data[(trie_index[(cp >> 6) as usize] + (cp & 63)) as usize];
        (p != 0).then(|| &decomp[(p >> 8) as usize..((p >> 8) + (p & 0xFF)) as usize])
    };
    let entry_char = |e: u32| char::from_u32(e & 0xFF_FFFF).unwrap();

    // self-validation: reconstruct from the trie == `decompose`, for every codepoint
    let bit = |cp: u32| (bitset[(cp / 64) as usize] >> (cp % 64)) & 1 != 0;
    for cp in 0..=0x10FFFFu32 {
        let Some(c) = char::from_u32(cp) else { continue };
        let expect = decompose(c);
        let got: Vec<char> = if cp < cap && bit(cp) {
            if HANGUL.contains(&cp) {
                let s = cp - 0xAC00;
                let mut v = vec![
                    char::from_u32(0x1100 + s / 588).unwrap(),
                    char::from_u32(0x1161 + (s % 588) / 28).unwrap(),
                ];
                if s % 28 != 0 {
                    v.push(char::from_u32(0x11A7 + s % 28).unwrap());
                }
                v
            } else {
                trie_lookup(cp)
                    .expect("unstable non-hangul cp missing from trie")
                    .iter()
                    .map(|&e| entry_char(e))
                    .collect()
            }
        } else {
            vec![c] // stable (or ≥ cap) → itself
        };
        assert_eq!(got, expect, "{form} table mismatch at cp={cp:#06x}");
        if cp < cap && bit(cp) && !HANGUL.contains(&cp) {
            for &e in trie_lookup(cp).unwrap() {
                assert_eq!((e >> 24) as u8, ccc(entry_char(e)), "{form} ccc mismatch in decomp of {cp:#x}");
            }
        }
    }

    let mut o = String::new();
    writeln!(o, "//! GENERATED by `bitmap_gen` — do NOT edit. Regenerate with `cargo run -p bitmap_gen`.").unwrap();
    writeln!(o, "//! {form} decomposition tables for `atomsplit::nfd`: stable-run bitset + a two-level trie").unwrap();
    writeln!(o, "//! (DATA[INDEX[cp>>6] + (cp&63)] = (off<<8)|len into DECOMP; 0 = absent) over a flattened").unwrap();
    writeln!(o, "//! decomposition blob. Reconstructed & checked against `unicode-normalization` for all 1.1M cps.\n").unwrap();
    writeln!(o, "/// Codepoints < this are covered by `{form}_UNSTABLE`; all higher ones are {form}-stable.").unwrap();
    writeln!(o, "pub const {form}_CAP: u32 = {cap:#x};").unwrap();
    writeln!(o, "/// Max UTF-8 byte expansion of any char under {form}: `out.len() ≤ input.len() * this`,").unwrap();
    writeln!(o, "/// so the runtime reserves a realloc-free output buffer.").unwrap();
    writeln!(o, "pub const {form}_MAX_EXPAND: usize = {max_expand};").unwrap();
    emit_slice(&mut o, "u64", &format!("{form}_UNSTABLE"), &bitset);
    emit_slice(&mut o, "u32", &format!("{form}_DECOMP"), &decomp);
    emit_slice(&mut o, "u32", &format!("{form}_TRIE_INDEX"), &trie_index);
    emit_slice(&mut o, "u32", &format!("{form}_TRIE_DATA"), &trie_data);
    o
}

/// NFD (canonical) decomposition tables.
pub fn generate_nfd_tables() -> String {
    use unicode_normalization::UnicodeNormalization;
    generate_decomp_tables("NFD", &|c| c.nfd().collect(), &nfd_unstable)
}
/// NFKD (compatibility) decomposition tables.
pub fn generate_nfkd_tables() -> String {
    use unicode_normalization::UnicodeNormalization;
    generate_decomp_tables("NFKD", &|c| c.nfkd().collect(), &nfkd_unstable)
}

// ── Composition tables (NFC + NFKC) ──────────────────────────────────────────────────────────────
//
// Composition combines a starter L with a following char C when (L, C) has a canonical primary
// composite and C is not "blocked". The pairwise table is the SAME for NFC and NFKC (only canonical
// composites ever recombine). Hangul L+V / LV+T are done by arithmetic at runtime, so they're excluded.
//   COMPOSE            sorted (key=(a<<21)|b, composite) — the (a,b) → composite lookup.
//   {NFC,NFKC}_RELEVANT  bitset: cp with ccc != 0 OR quick-check != Yes. If a string trips none of these
//                        it is already in the target form → the kernel borrows it untouched.

/// Composition table + NFC/NFKC relevance bitsets, self-validated against `unicode-normalization`.
pub fn generate_compose_tables() -> String {
    use unicode_normalization::char::{canonical_combining_class as ccc, compose};
    use unicode_normalization::{is_nfc_quick, is_nfkc_quick, IsNormalized, UnicodeNormalization};

    // Primary (single-step) canonical composites. For a char X with a canonical decomposition, the
    // single-step decomposition is [A, B] where B is the LAST decomposed char and A is the recomposition
    // of the prefix (NFC of the prefix → one char). This recovers NESTED composites (e.g. U+1EA4 Ấ →
    // (U+00C2 Â, U+0301)) that a naive `nfd(X) == [a,b]` test misses — Ấ's FULL NFD is length 3. Hangul is
    // arithmetic (excluded); `compose` filters composition exclusions & non-starter decompositions.
    let single_step = |c: char| -> Option<(char, char)> {
        let full: Vec<char> = c.nfd().collect();
        if full.len() < 2 {
            return None; // no decomposition, or a singleton (decompose-only, never composes)
        }
        let b = *full.last().unwrap();
        let prefix: String = full[..full.len() - 1].iter().collect();
        let mut a = prefix.nfc();
        let a0 = a.next()?;
        if a.next().is_some() {
            return None; // prefix didn't recompose to a single char → not a primary composite
        }
        Some((a0, b))
    };

    let mut table: Vec<(u64, u32)> = Vec::new();
    for cp in 0..0x110000u32 {
        let Some(c) = char::from_u32(cp) else { continue };
        if HANGUL.contains(&cp) {
            continue;
        }
        if let Some((a, b)) = single_step(c) {
            if compose(a, b) == Some(c) {
                table.push((((a as u64) << 21) | b as u64, cp));
            }
        }
    }
    table.sort_by_key(|&(k, _)| k);
    for w in table.windows(2) {
        assert_ne!(w[0].0, w[1].0, "duplicate compose key");
    }

    // Hangul jamo composition, arithmetic (mirrors the runtime kernel).
    let hangul_compose = |a: u32, b: u32| -> Option<u32> {
        if (0x1100..=0x1112).contains(&a) && (0x1161..=0x1175).contains(&b) {
            return Some(0xAC00 + ((a - 0x1100) * 21 + (b - 0x1161)) * 28);
        }
        if HANGUL.contains(&a) && (a - 0xAC00) % 28 == 0 && (0x11A8..=0x11C2).contains(&b) {
            return Some(a + (b - 0x11A7));
        }
        None
    };
    let lookup = |a: u32, b: u32| -> Option<u32> {
        let k = ((a as u64) << 21) | b as u64;
        table.binary_search_by_key(&k, |&(kk, _)| kk).ok().map(|p| table[p].1)
    };
    // validate completeness: for every char's single-step pair, table (or Hangul arithmetic) == compose()
    for cp in 0..0x110000u32 {
        let Some(c) = char::from_u32(cp) else { continue };
        if let Some((a, b)) = single_step(c) {
            let expect = compose(a, b).map(|x| x as u32);
            let got = hangul_compose(a as u32, b as u32).or_else(|| lookup(a as u32, b as u32));
            assert_eq!(got, expect, "compose mismatch for ({:#x},{:#x})", a as u32, b as u32);
        }
    }

    let nfc_relevant = |c: char| {
        ccc(c) != 0 || !matches!(is_nfc_quick(std::iter::once(c)), IsNormalized::Yes)
    };
    let nfkc_relevant = |c: char| {
        ccc(c) != 0 || !matches!(is_nfkc_quick(std::iter::once(c)), IsNormalized::Yes)
    };
    let (nfc_cap, nfc_bits) = bitset_up_to(&nfc_relevant);
    let (nfkc_cap, nfkc_bits) = bitset_up_to(&nfkc_relevant);

    let mut o = String::new();
    writeln!(o, "//! GENERATED by `bitmap_gen` — do NOT edit. Regenerate with `cargo run -p bitmap_gen`.").unwrap();
    writeln!(o, "//! Canonical composition table (NFC/NFKC) + quick-check relevance bitsets. `bitmap_gen`").unwrap();
    writeln!(o, "//! validates COMPOSE against `unicode-normalization`'s `compose` for every canonical pair.\n").unwrap();
    write!(o, "/// Sorted `(key=(a<<21)|b, composite)`; `a`,`b` are the canonical decomposition pair.\n").unwrap();
    write!(o, "#[rustfmt::skip]\npub static COMPOSE: [(u64, u32); {}] = [", table.len()).unwrap();
    for (i, (k, x)) in table.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        write!(o, "({k},0x{x:X})").unwrap();
    }
    o.push_str("];\n");
    writeln!(o, "/// A codepoint < this may be composition-relevant; higher ones never are (NFC).").unwrap();
    writeln!(o, "pub const NFC_RELEVANT_CAP: u32 = {nfc_cap:#x};").unwrap();
    emit_slice(&mut o, "u64", "NFC_RELEVANT", &nfc_bits);
    writeln!(o, "/// A codepoint < this may be composition-relevant; higher ones never are (NFKC).").unwrap();
    writeln!(o, "pub const NFKC_RELEVANT_CAP: u32 = {nfkc_cap:#x};").unwrap();
    emit_slice(&mut o, "u64", "NFKC_RELEVANT", &nfkc_bits);
    o
}
