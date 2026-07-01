use std::sync::LazyLock;

use regex::Regex;

use crate::pipeline::{self, SplitPolicy};
use crate::tokenizer::{
    pattern::Invert, PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior,
};
use crate::utils::macro_rules_attribute;

#[derive(Clone, Debug, PartialEq, Eq)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct Whitespace;

impl Default for Whitespace {
    fn default() -> Self {
        Self
    }
}

impl PreTokenizer for Whitespace {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\w+|[^\w\s]+").unwrap());
        let re_ref: &Regex = &RE;

        pretokenized.split(|_, normalized| {
            normalized.split(Invert(re_ref), SplitDelimiterBehavior::Removed)
        })
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct WhitespaceSplit;

impl PreTokenizer for WhitespaceSplit {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, normalized| {
            normalized.split(char::is_whitespace, SplitDelimiterBehavior::Removed)
        })
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum CharType {
    Whitespace,
    Word,
    Symbol,
}

impl CharType {
    #[inline]
    fn policy(self) -> SplitPolicy {
        match self {
            // whitespace is dropped; word and symbol groups are each emitted as one
            // split, with a boundary between the two classes.
            CharType::Whitespace => SplitPolicy::Remove,
            CharType::Word | CharType::Symbol => SplitPolicy::Keep,
        }
    }
}

fn classify(ch: char) -> CharType {
    if ch.is_whitespace() {
        CharType::Whitespace
    } else if is_word_char(ch) {
        CharType::Word
    } else {
        CharType::Symbol
    }
}

/// Matches the same characters as the `\w` regex class (Unicode-aware).
/// This is: Alphabetic + Nd (decimal digit) + Pc (connector punctuation) +
/// M (marks) + Join_Control — NOT Nl/No (which Rust's is_alphanumeric includes).
fn is_word_char(ch: char) -> bool {
    use unicode_categories::UnicodeCategories;

    ch.is_alphabetic()
        || ch.is_number_decimal_digit()
        || ch.is_punctuation_connector()
        || ch.is_mark()
        || ch == '\u{200c}' // Zero-Width Non-Joiner
        || ch == '\u{200d}' // Zero-Width Joiner
}

static ASCII_CLASS: LazyLock<[CharType; 128]> =
    LazyLock::new(|| std::array::from_fn(|b| classify(b as u8 as char)));

impl pipeline::PreTokenizer for Whitespace {
    // XXX: surprisingly, inlining here yields 10-15% slower performance
    #[inline(never)]
    fn pre_tokenize(&self, text: &str, out: &mut Vec<pipeline::Split>) -> Result<()> {
        pipeline::split(
            text,
            out,
            |ch| {
                if ch.is_ascii() {
                    ASCII_CLASS[ch as usize]
                } else {
                    classify(ch)
                }
            },
            CharType::policy,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer};

    fn pretokenize(text: &str) -> Vec<(&str, (u32, u32))> {
        let pretok = Whitespace;
        let mut splits = Vec::new();
        crate::pipeline::PreTokenizer::pre_tokenize(&pretok, text, &mut splits).unwrap();
        splits
            .iter()
            .map(|s| (&text[s.range()], (s.start, s.end)))
            .collect()
    }

    #[test]
    fn basic() {
        let tests = vec![
            (
                "Hey man!",
                vec![("Hey", (0, 3)), ("man", (4, 7)), ("!", (7, 8))],
            ),
            (
                "How are you doing?",
                vec![
                    ("How", (0, 3)),
                    ("are", (4, 7)),
                    ("you", (8, 11)),
                    ("doing", (12, 17)),
                    ("?", (17, 18)),
                ],
            ),
            ("\n", vec![]),
        ];
        for (s, res) in tests {
            assert_eq!(pretokenize(s), res, "input: {s:?}");
        }
    }

    #[test]
    fn whitespace_split() {
        let tests = vec![
            ("Hey man!", vec![("Hey", (0, 3)), ("man!", (4, 8))]),
            (
                "Hey, man, Good?",
                vec![("Hey,", (0, 4)), ("man,", (5, 9)), ("Good?", (10, 15))],
            ),
        ];
        let pretok = WhitespaceSplit;
        for (s, res) in tests {
            let mut pretokenized = PreTokenizedString::from(s);
            pretok.pre_tokenize(&mut pretokenized).unwrap();
            assert_eq!(
                pretokenized
                    .get_splits(OffsetReferential::Original, OffsetType::Byte)
                    .into_iter()
                    .map(|(s, o, _)| (s, o))
                    .collect::<Vec<_>>(),
                res
            );
        }
    }

    #[test]
    fn edge_cases() {
        // word / symbol / whitespace transitions; whitespace dropped, splits kept whole
        let edge_cases = vec![
            ("", vec![]),
            (" ", vec![]),
            ("  ", vec![]),
            ("a", vec![("a", (0, 1))]),
            ("!", vec![("!", (0, 1))]),
            ("a!", vec![("a", (0, 1)), ("!", (1, 2))]),
            ("!a", vec![("!", (0, 1)), ("a", (1, 2))]),
            ("a b", vec![("a", (0, 1)), ("b", (2, 3))]),
            ("a  b", vec![("a", (0, 1)), ("b", (3, 4))]),
            ("a\tb", vec![("a", (0, 1)), ("b", (2, 3))]),
            ("a\nb", vec![("a", (0, 1)), ("b", (2, 3))]),
            ("a\r\nb", vec![("a", (0, 1)), ("b", (3, 4))]),
        ];

        for (input, expected) in edge_cases {
            assert_eq!(pretokenize(input), expected, "input: {input:?}");
        }
    }

    #[test]
    fn multibyte_offsets() {
        // offsets are byte offsets into the input; classification is Unicode-aware.
        // é is 2 bytes, so "café" = 0..5 and "résumé" = 6..14.
        assert_eq!(
            pretokenize("café résumé"),
            vec![("café", (0, 5)), ("résumé", (6, 14))],
        );
        // CJK ideographs are alphabetic (word chars): one split, no inner boundary.
        assert_eq!(
            pretokenize("中文 text"),
            vec![("中文", (0, 6)), ("text", (7, 11))],
        );
        // '_' is connector punctuation (a word char) -> a single word token.
        assert_eq!(pretokenize("hello_world"), vec![("hello_world", (0, 11))]);
        // word and symbol groups are each one split; only the boundary splits.
        assert_eq!(
            pretokenize("ab!!cd"),
            vec![("ab", (0, 2)), ("!!", (2, 4)), ("cd", (4, 6))],
        );
    }

    // TODO: add xnli test:
    // - either as an integration test
    // - either as a unit test that triggers only if the xnli file is present in the data/ dir
    // #[test]
    // fn xnli() {}
}
