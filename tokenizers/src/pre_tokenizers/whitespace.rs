use std::sync::LazyLock;

use regex::Regex;

use crate::pattern::Pattern;
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct ManualWhitespaceSplit;

impl PreTokenizer for ManualWhitespaceSplit {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, normalized| {
            normalized.split(WhiteSpacePattern, SplitDelimiterBehavior::Removed)
        })
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum CharType {
    Whitespace,
    Word,
    Symbol,
}

struct WhiteSpacePattern;

impl Pattern for WhiteSpacePattern {
    fn find_matches(&self, inside: &str) -> Result<Vec<(crate::Offsets, bool)>> {
        if inside.is_empty() {
            return Ok(vec![((0, 0), false)]);
        }

        let mut matches = Vec::new();
        let mut span_start = 0;
        let mut prev_type: Option<CharType> = None;

        for (i, ch) in inside.char_indices() {
            let ct = classify(ch);

            if let Some(pt) = prev_type {
                if pt != ct {
                    // Emit the previous span:
                    // - whitespace spans are non-matches (false)
                    // - word/symbol spans are matches (true)
                    matches.push(((span_start, i), pt == CharType::Whitespace));
                    span_start = i;
                }
            }
            prev_type = Some(ct);
        }

        // Emit the final span
        if let Some(pt) = prev_type {
            matches.push(((span_start, inside.len()), pt == CharType::Whitespace));
        }

        Ok(matches)
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

    ch.is_alphabetic() // Unicode Alphabetic property (L* + some others)
        || ch.is_number_decimal_digit() // Nd only (not Nl/No like superscripts, fractions)
        || ch.is_punctuation_connector() // Pc: underscore, undertie, fullwidth low line, etc.
        || ch.is_mark_nonspacing() // Mn: combining diacriticals, nukta, etc.
        || ch.is_mark_spacing_combining() // Mc: spacing combining marks (vowel signs)
        || ch.is_mark_enclosing() // Me: enclosing marks
        || ch == '\u{200c}' // Zero-Width Non-Joiner
        || ch == '\u{200d}' // Zero-Width Joiner
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OffsetReferential, OffsetType, PreTokenizer};

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
        let pretok = Whitespace {};
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
    fn assert_equivalent() {
        let test_cases = vec![
            "Hello world!",
            "How are you doing?",
            "This is a test with numbers 123 and symbols @#$%",
            "Multiple    spaces",
            "Tabs\tand\nnewlines",
            "Unicode: café résumé naïve",
            "Mixed: Hello123!@# world",
            "Edge cases: a.b,c;d:e",
            "Empty string:",
            "Only spaces:   ",
            "Only symbols: !@#$%",
            "Only words: hello world",
            "Numbers: 123 456 789",
            "Underscores: hello_world test_case",
            "Special chars: αβγ δέζ ηθι",
        ];

        for test_case in test_cases {
            let mut original = PreTokenizedString::from(test_case);
            let mut manual = PreTokenizedString::from(test_case);

            let original_pretok = Whitespace {};
            let manual_pretok = ManualWhitespaceSplit {};

            original_pretok.pre_tokenize(&mut original).unwrap();
            manual_pretok.pre_tokenize(&mut manual).unwrap();

            let original_splits = original
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>();

            let manual_splits = manual
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>();

            assert_eq!(
                original_splits, manual_splits,
                "Mismatch for test case: '{}'",
                test_case
            );
        }
    }

    #[test]
    fn manual_edge_cases() {
        let pretok = ManualWhitespaceSplit {};

        // Test various edge cases
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
            let mut pretokenized = PreTokenizedString::from(input);
            pretok.pre_tokenize(&mut pretokenized).unwrap();
            let result = pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>();
            assert_eq!(result, expected, "Failed for input: '{}'", input);
        }
    }

    #[test]
    fn assert_equivalent_xnli() {
        let data = std::fs::read_to_string("data/xnli.txt").unwrap();
        let original_pretok = Whitespace {};
        let manual_pretok = ManualWhitespaceSplit {};

        for (i, line) in data.lines().enumerate() {
            let mut original = PreTokenizedString::from(line);
            let mut manual = PreTokenizedString::from(line);

            original_pretok.pre_tokenize(&mut original).unwrap();
            manual_pretok.pre_tokenize(&mut manual).unwrap();

            let original_splits = original
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>();
            let manual_splits = manual
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>();

            assert_eq!(
                original_splits,
                manual_splits,
                "Mismatch on line {}: '{}'",
                i,
                &line.chars().take(80).collect::<String>(),
            );
        }
    }
}
