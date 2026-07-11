use std::borrow::Cow;
use std::cell::RefCell;

use crate::{
    pipeline,
    tokenizer::{NormalizedString, Normalizer, Result},
};

use atomsplit::classify::classify;
use atomsplit::norm_classify::NormClass;
use serde::{Deserialize, Serialize};
use unicode_categories::UnicodeCategories;

thread_local! {
    /// Per-thread scratch for the classifier's per-byte tag stream — reused across calls so the fast
    /// `Normalizer::normalize` path is zero-alloc after warmup (per-thread → parallel encode stays lock-free).
    static TAGS: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Checks whether a character is whitespace
fn is_whitespace(c: char) -> bool {
    // These are technically control characters but we count them as whitespace
    match c {
        '\t' | '\n' | '\r' => true,
        _ => c.is_whitespace(),
    }
}

/// Checks whether a character is a control character
fn is_control(c: char) -> bool {
    // These are technically control characters but we count them as whitespace
    match c {
        '\t' | '\n' | '\r' => false,
        // The definition of `is_control` here is quite large and contains also
        // Cc, Cf, Cn or Co
        // cf. https://unicode.org/reports/tr44/ (Table 12)
        _ => c.is_other(),
    }
}

/// Whether BERT text cleaning removes `c` entirely
fn clean_text_removes(c: char) -> bool {
    c == '\0' || c == '\u{fffd}' || is_control(c)
}

/// The whitespace folding BERT text cleaning applies to kept characters
fn clean_text_map(c: char) -> char {
    if is_whitespace(c) {
        ' '
    } else {
        c
    }
}

/// Checks whether a character is chinese
/// This defines a "chinese character" as anything in the CJK Unicode block:
///   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
///
/// Note that the CJK Unicode block is NOT all Japanese and Korean characters,
/// despite its name. The modern Korean Hangul alphabet is a different block,
/// as is Japanese Hiragana and Katakana. Those alphabets are used to write
/// space-separated words, so they are not treated specially and handled
/// like for all of the other languages.
fn is_chinese_char(c: char) -> bool {
    matches!(
        c as usize,
        0x4E00..=0x9FFF |
        0x3400..=0x4DBF |
        0x20000..=0x2A6DF |
        0x2A700..=0x2B73F |
        0x2B740..=0x2B81F |
        0x2B920..=0x2CEAF |
        0xF900..=0xFAFF |
        0x2F800..=0x2FA1F
    )
}

#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub struct BertNormalizer {
    /// Whether to do the bert basic cleaning:
    ///   1. Remove any control characters
    ///   2. Replace all sorts of whitespace by the classic one ` `
    pub clean_text: bool,
    /// Whether to put spaces around chinese characters so they get split
    pub handle_chinese_chars: bool,
    /// Whether to strip accents
    pub strip_accents: Option<bool>,
    /// Whether to lowercase the input
    pub lowercase: bool,
}

impl Default for BertNormalizer {
    fn default() -> Self {
        Self {
            clean_text: true,
            handle_chinese_chars: true,
            strip_accents: None,
            lowercase: true,
        }
    }
}

impl BertNormalizer {
    pub fn new(
        clean_text: bool,
        handle_chinese_chars: bool,
        strip_accents: Option<bool>,
        lowercase: bool,
    ) -> Self {
        Self {
            clean_text,
            handle_chinese_chars,
            strip_accents,
            lowercase,
        }
    }

    fn do_clean_text(&self, normalized: &mut NormalizedString) {
        normalized
            .filter(|c| !clean_text_removes(c))
            .map(clean_text_map);
    }

    fn do_handle_chinese_chars(&self, normalized: &mut NormalizedString) {
        let mut new_chars: Vec<(char, isize)> = vec![];
        normalized.for_each(|c| {
            if is_chinese_char(c) {
                new_chars.extend([(' ', 0), (c, 1), (' ', 1)]);
            } else {
                new_chars.push((c, 0));
            }
        });
        normalized.transform(new_chars, 0);
    }

    fn do_strip_accents(&self, normalized: &mut NormalizedString) {
        normalized.nfd().filter(|c| !c.is_mark_nonspacing());
    }

    fn do_lowercase(&self, normalized: &mut NormalizedString) {
        normalized.lowercase();
    }
}

impl Normalizer for BertNormalizer {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        if self.clean_text {
            self.do_clean_text(normalized);
        }
        if self.handle_chinese_chars {
            self.do_handle_chinese_chars(normalized);
        }
        let strip_accents = self.strip_accents.unwrap_or(self.lowercase);
        if strip_accents {
            self.do_strip_accents(normalized);
        }
        if self.lowercase {
            self.do_lowercase(normalized);
        }

        Ok(())
    }
}

impl BertNormalizer {
    /// Which `norm_classify` bits make a char non-inert under the active rules — i.e. a char with none
    /// of these bits set is left unchanged by every enabled stage and can be copied verbatim.
    fn active_mask(&self, strip_accents: bool) -> u8 {
        use atomsplit::norm_classify::bit;
        (if self.clean_text { bit::CTRL | bit::WS } else { 0 })
            | (if self.handle_chinese_chars { bit::CJK } else { 0 })
            // bert `strip_accents` runs canonical NFD then drops Mn; the NFD bit already folds in
            // reorderable (ccc != 0) chars, so no separate reorder bit is needed.
            | (if strip_accents { bit::NFD | bit::MARK } else { 0 })
            | (if self.lowercase { bit::LOWER } else { 0 })
    }

    /// Tag-driven normalize. `tags` is `input`'s per-byte classification (`tags.len() == input.len()`,
    /// char-start bytes carry the [`atomsplit::norm_classify`] bitmask; the caller produced it with the
    /// SIMD or scalar classifier). Maximal **inert** runs (no active bit) are copied verbatim; only the
    /// runs that actually change go through per-char dispatch ([`Self::dispatch_char`]). Byte-exact with
    /// the legacy path: inert chars are starters (ccc 0 — reorderables carry the NFD bit), so an NFD
    /// reorder never crosses a run boundary.
    pub fn normalize_from_tags<'a>(&self, input: &'a str, tags: &[u8]) -> Cow<'a, str> {
        use atomsplit::classify::char_len;
        let strip_accents = self.strip_accents.unwrap_or(self.lowercase);
        let active = self.active_mask(strip_accents);
        let bytes = input.as_bytes();
        let n = bytes.len();

        // Longest inert prefix is borrowable; if the whole string is inert, return it untouched.
        let mut first = 0;
        while first < n && tags[first] & active == 0 {
            first += char_len(bytes[first]);
        }
        if first == n {
            return Cow::Borrowed(input);
        }

        let mut out = String::with_capacity(n);
        out.push_str(&input[..first]);
        let mut i = first;
        while i < n {
            // non-inert run → per-char dispatch (each rule runs only where its bit is set)
            let ns = i;
            while i < n && tags[i] & active != 0 {
                i += char_len(bytes[i]);
            }
            for (off, c) in input[ns..i].char_indices() {
                self.dispatch_char(c, tags[ns + off], strip_accents, &mut out);
            }
            // inert run → verbatim
            let is = i;
            while i < n && tags[i] & active == 0 {
                i += char_len(bytes[i]);
            }
            out.push_str(&input[is..i]);
        }
        Cow::Owned(out)
    }

    /// Normalize one char from its `norm_classify` tag: run each rule ONLY where its bit is set. A
    /// script with no cased chars never calls `to_lowercase`; a char that doesn't decompose never pays
    /// a `canonical_combining_class` lookup. Order matches the pipeline: clean → chinese → the transform.
    #[inline]
    fn dispatch_char(&self, c: char, tg: u8, strip_accents: bool, out: &mut String) {
        use atomsplit::norm_classify::bit;
        if self.clean_text {
            if tg & bit::CTRL != 0 {
                return; // removed
            }
            if tg & bit::WS != 0 {
                out.push(' '); // whitespace folds to space (which carries no further rule)
                return;
            }
        }
        if self.handle_chinese_chars && tg & bit::CJK != 0 {
            out.push(' ');
            self.emit_transform(c, tg, strip_accents, out);
            out.push(' ');
            return;
        }
        self.emit_transform(c, tg, strip_accents, out);
    }

    /// NFD → strip nonspacing marks → lowercase, applied per the char's bits. Decomposition runs (via the
    /// pure-Rust `atomsplit::nfd::nfd_char`) only on NFD-flagged chars; a mark is dropped, anything else is
    /// a `to_lowercase` (LOWER) or a plain push. Byte-exact with the run-based pipeline: strip drops every
    /// Mn, so per-char decomposition needs no cross-char reordering.
    #[inline]
    fn emit_transform(&self, c: char, tg: u8, strip_accents: bool, out: &mut String) {
        use atomsplit::norm_classify::bit;
        if strip_accents && tg & bit::NFD != 0 {
            // Pure-Rust NFD decomposition (baked trie + arithmetic Hangul, no `unicode-normalization`):
            // decompose the char, drop nonspacing marks (bert's strip), lowercase the rest if enabled.
            atomsplit::nfd::nfd_char(c, |d| {
                if d.is_mark_nonspacing() {
                    return;
                }
                if self.lowercase {
                    out.extend(d.to_lowercase());
                } else {
                    out.push(d);
                }
            });
        } else if strip_accents && tg & bit::MARK != 0 {
            // MARK is any combining mark; bert strips only nonspacing marks (Mn). Keep the rest
            // (e.g. spacing marks), lowercasing if enabled — matches `nfd().filter(!Mn)`.
            if !c.is_mark_nonspacing() {
                if self.lowercase {
                    out.extend(c.to_lowercase());
                } else {
                    out.push(c);
                }
            }
        } else if self.lowercase && tg & bit::LOWER != 0 {
            out.extend(c.to_lowercase());
        } else {
            out.push(c);
        }
    }
}

impl pipeline::Normalizer for BertNormalizer {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
        if input.is_empty() {
            return Ok(Cow::Borrowed(input));
        }
        // One SIMD pass tags every byte; `normalize_from_tags` then copies inert runs verbatim and runs
        // each rule only where its bit is set. `Cow::Borrowed` is returned untouched when nothing changes.
        TAGS.with(|cell| {
            let mut tags = cell.borrow_mut();
            tags.clear();
            tags.resize(input.len(), 0);
            classify::<NormClass>(input.as_bytes(), &mut tags);
            // The returned Cow borrows `input` (or owns a fresh String); it never borrows `tags`.
            Ok(match self.normalize_from_tags(input, &tags) {
                Cow::Borrowed(s) => Cow::Borrowed(s),
                Cow::Owned(s) => Cow::Owned(s),
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INPUTS: &[&str] = &[
        "Héllo World",
        "中文字",
        "a中b文c",
        "  spaced  ",
        "abc",
        "",
        "\tTab\n\r",
        "MiXeD Café",
        "e\u{0301}",        // already-NFD combining acute
        "\u{fb01}ligature", // NFKC ligature (unchanged by NFD)
        "null\0here",
        "repl\u{fffd}char",
        "ctrl\u{0007}char",
        "ǅ",        // titlecase, lowercases to multi-mapping
        "İstanbul", // dotted capital I: lowercases to 2 chars
        "straße",
        // run-split hardening: marks that NFD-reorders (ccc 230 then 220), on ASCII and non-ASCII bases,
        // across the ASCII/non-ASCII boundary; non-ASCII whitespace folded to space; a format char
        // (zero-width space, Cf) that clean_text removes; and a long mixed-script line.
        "e\u{0301}\u{0323}",
        "e\u{0323}\u{0301}",
        "\u{00e8}\u{0323}\u{0301}",
        "a\u{00a0}b\u{2028}c",
        "a\u{200b}b",
        "The 世界 Café tëst\u{0301} 123 ǅ Москва",
        // Hangul: exercises the arithmetic S_BASE decompose — 한/국 have a trailing jamo (3), 어 has
        // none (2); 가 = first syllable (U+AC00), 힣 = last (U+D7A3).
        "한국어 안녕 가힣",
        // Devanagari: spacing combining marks (Mc, e.g. vowel signs) are `is_combining_mark` but NOT Mn,
        // so bert must KEEP them (only Mn stripped) — exercises the MARK-branch runtime refine.
        "नमस्ते दुनिया",
    ];

    #[test]
    fn pipeline_bert_matches_legacy() {
        for &clean_text in &[true, false] {
            for &handle_chinese_chars in &[true, false] {
                for &strip_accents in &[None, Some(true), Some(false)] {
                    for &lowercase in &[true, false] {
                        let n = BertNormalizer::new(
                            clean_text,
                            handle_chinese_chars,
                            strip_accents,
                            lowercase,
                        );
                        for input in INPUTS {
                            let mut ns = NormalizedString::from(*input);
                            Normalizer::normalize(&n, &mut ns).unwrap(); // legacy oracle
                            assert_eq!(
                                ns.get(),
                                &*pipeline::Normalizer::normalize(&n, input).unwrap(),
                                "config={n:?} input={input:?}",
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn tag_driven_bert_matches_legacy() {
        use atomsplit::classify::{classify, classify_scalar};
        use atomsplit::norm_classify::NormClass;
        for &clean_text in &[true, false] {
            for &handle_chinese_chars in &[true, false] {
                for &strip_accents in &[None, Some(true), Some(false)] {
                    for &lowercase in &[true, false] {
                        let n = BertNormalizer::new(
                            clean_text,
                            handle_chinese_chars,
                            strip_accents,
                            lowercase,
                        );
                        for input in INPUTS {
                            let mut ns = NormalizedString::from(*input);
                            Normalizer::normalize(&n, &mut ns).unwrap(); // legacy oracle
                            let expected = ns.get();

                            let mut scal = vec![0u8; input.len()];
                            classify_scalar::<NormClass>(input.as_bytes(), &mut scal);
                            let mut simd = vec![0u8; input.len()];
                            classify::<NormClass>(input.as_bytes(), &mut simd);
                            assert_eq!(simd, scal, "SIMD/scalar tags differ on {input:?}");

                            assert_eq!(
                                &*n.normalize_from_tags(input, &scal),
                                expected,
                                "tag-driven config={n:?} input={input:?}",
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn pipeline_bert_borrows_when_noop() {
        let n = BertNormalizer::default();
        for input in &["hello world", "already lowercase ascii", ""] {
            assert!(matches!(
                pipeline::Normalizer::normalize(&n, input).unwrap(),
                Cow::Borrowed(_)
            ));
        }
        // Anything that must change is owned.
        for input in &["Héllo", "中", "\tx", "ABC"] {
            assert!(matches!(
                pipeline::Normalizer::normalize(&n, input).unwrap(),
                Cow::Owned(_)
            ));
        }
    }
}
