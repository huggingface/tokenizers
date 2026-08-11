use std::borrow::Cow;

use crate::{
    pipeline,
    tokenizer::{Normalizer, Result},
};

use super::utils::lowercases_to_self;

use serde::{Deserialize, Serialize};
use unicode_categories::UnicodeCategories;
use unicode_normalization::{IsNormalized, UnicodeNormalization, is_nfd_quick};

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
    if is_whitespace(c) { ' ' } else { c }
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

    fn is_noop(&self, input: &str, strip_accents: bool) -> bool {
        if strip_accents && !matches!(is_nfd_quick(input.chars()), IsNormalized::Yes) {
            return false;
        }
        let changes = |c: char| {
            (self.clean_text && (clean_text_removes(c) || clean_text_map(c) != c))
                || (self.handle_chinese_chars && is_chinese_char(c))
                || (strip_accents && c.is_mark_nonspacing())
                || (self.lowercase && !lowercases_to_self(c))
        };
        !input.chars().any(changes)
    }
}

impl Normalizer for BertNormalizer {}

impl pipeline::Normalizer for BertNormalizer {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
        let strip_accents = self.strip_accents.unwrap_or(self.lowercase);

        if self.is_noop(input, strip_accents) {
            return Ok(input.into());
        }

        let cleaned = input
            .chars()
            .filter(|&c| !(self.clean_text && clean_text_removes(c)))
            .flat_map(|c| {
                let c = if self.clean_text {
                    clean_text_map(c)
                } else {
                    c
                };
                if self.handle_chinese_chars && is_chinese_char(c) {
                    [Some(' '), Some(c), Some(' ')]
                } else {
                    [Some(c), None, None]
                }
            })
            .flatten();

        // `.nfd()` changes the iterator's type, so the stage toggles can't be
        // plain `if`s mid-chain: each config combination gets its own
        // statically-typed chain, all funneled into one pre-sized String.
        let mut normalized = String::with_capacity(input.len());
        match (strip_accents, self.lowercase) {
            (true, true) => normalized.extend(
                cleaned
                    .nfd()
                    .filter(|c| !c.is_mark_nonspacing())
                    .flat_map(char::to_lowercase),
            ),
            (true, false) => normalized.extend(cleaned.nfd().filter(|c| !c.is_mark_nonspacing())),
            (false, true) => normalized.extend(cleaned.flat_map(char::to_lowercase)),
            (false, false) => normalized.extend(cleaned),
        }

        Ok(Cow::Owned(normalized))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalizers::assert_normalizes;

    /// The inputs every config below is checked against: one per stage this normalizer runs,
    /// plus the characters whose lowercase mapping is not a single identical char.
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
    ];

    /// Pairs `INPUTS` with the output each one is expected to produce.
    fn cases(expected: &'static [&'static str]) -> Vec<(&'static str, &'static str)> {
        assert_eq!(expected.len(), INPUTS.len());
        INPUTS
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .collect()
    }

    #[test]
    fn every_stage_off_leaves_the_input_alone() {
        let n = BertNormalizer::new(false, false, Some(false), false);
        assert_normalizes(&n, &cases(INPUTS));
    }

    #[test]
    fn default_config_runs_every_stage() {
        assert_normalizes(
            &BertNormalizer::default(),
            &cases(&[
                "hello world",
                " 中  文  字 ",
                "a 中 b 文 c",
                "  spaced  ",
                "abc",
                "",
                " tab  ",
                "mixed cafe",
                "e",
                "ﬁligature",
                "nullhere",
                "replchar",
                "ctrlchar",
                "ǆ",
                "istanbul",
                "straße",
            ]),
        );
    }

    #[test]
    fn clean_text_folds_whitespace_and_drops_controls() {
        let n = BertNormalizer::new(true, false, Some(false), false);
        assert_normalizes(
            &n,
            &cases(&[
                "Héllo World",
                "中文字",
                "a中b文c",
                "  spaced  ",
                "abc",
                "",
                " Tab  ",
                "MiXeD Café",
                "e\u{0301}",
                "ﬁligature",
                "nullhere",
                "replchar",
                "ctrlchar",
                "ǅ",
                "İstanbul",
                "straße",
            ]),
        );
    }

    #[test]
    fn handle_chinese_chars_pads_cjk_with_spaces() {
        let n = BertNormalizer::new(false, true, Some(false), false);
        assert_normalizes(
            &n,
            &cases(&[
                "Héllo World",
                " 中  文  字 ",
                "a 中 b 文 c",
                "  spaced  ",
                "abc",
                "",
                "\tTab\n\r",
                "MiXeD Café",
                "e\u{0301}",
                "ﬁligature",
                "null\0here",
                "repl\u{fffd}char",
                "ctrl\u{0007}char",
                "ǅ",
                "İstanbul",
                "straße",
            ]),
        );
    }

    #[test]
    fn strip_accents_decomposes_then_drops_marks() {
        let n = BertNormalizer::new(false, false, Some(true), false);
        assert_normalizes(
            &n,
            &cases(&[
                "Hello World",
                "中文字",
                "a中b文c",
                "  spaced  ",
                "abc",
                "",
                "\tTab\n\r",
                "MiXeD Cafe",
                "e",
                "ﬁligature",
                "null\0here",
                "repl\u{fffd}char",
                "ctrl\u{0007}char",
                "ǅ",
                // NFD splits the dot above off "İ", and the dot is a mark
                "Istanbul",
                "straße",
            ]),
        );
    }

    #[test]
    fn lowercase_folds_case() {
        let n = BertNormalizer::new(false, false, Some(false), true);
        assert_normalizes(
            &n,
            &cases(&[
                "héllo world",
                "中文字",
                "a中b文c",
                "  spaced  ",
                "abc",
                "",
                "\ttab\n\r",
                "mixed café",
                "e\u{0301}",
                "ﬁligature",
                "null\0here",
                "repl\u{fffd}char",
                "ctrl\u{0007}char",
                "ǆ",
                // Without accent stripping, "İ" lowercases to "i" plus a combining dot
                "i\u{307}stanbul",
                "straße",
            ]),
        );
    }

    #[test]
    fn unset_strip_accents_follows_lowercase() {
        let stripped = &[("Héllo", "hello"), ("MiXeD Café", "mixed cafe")];
        assert_normalizes(&BertNormalizer::new(false, false, None, true), stripped);
        assert_normalizes(
            &BertNormalizer::new(false, false, Some(true), true),
            stripped,
        );

        let kept = &[("Héllo", "Héllo"), ("MiXeD Café", "MiXeD Café")];
        assert_normalizes(&BertNormalizer::new(false, false, None, false), kept);
        assert_normalizes(&BertNormalizer::new(false, false, Some(false), false), kept);
    }

    #[test]
    fn borrows_the_input_when_nothing_changes() {
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
