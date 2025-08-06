use std::sync::LazyLock;

use regex::Regex;

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

/// Optimized whitespace pre-tokenizer that uses byte-level scanning instead of regex.
/// This provides better performance but may have slightly different behavior in edge cases
/// compared to the regex-based implementation.
#[derive(Clone, Debug, PartialEq, Eq)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct WhitespaceOptimized;

impl Default for WhitespaceOptimized {
    fn default() -> Self {
        Self
    }
}

impl PreTokenizer for WhitespaceOptimized {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, normalized| {
            normalized.split(Invert(WhitespacePattern), SplitDelimiterBehavior::Removed)
        })
    }
}

/// Custom pattern implementation for optimized whitespace splitting
/// This implements the equivalent of the regex r"\w+|[^\w\s]+" but with manual byte scanning
struct WhitespacePattern;

impl crate::tokenizer::pattern::Pattern for WhitespacePattern {
    fn find_matches(&self, inside: &str) -> Result<Vec<(crate::Offsets, bool)>> {
        if inside.is_empty() {
            return Ok(vec![((0, 0), false)]);
        }

        let mut matches = Vec::new();
        let mut current_start = 0;
        let mut current_end = 0;
        let mut current_type = None; // None = whitespace, Some(true) = word, Some(false) = symbol

        let mut i = 0;
        while i < inside.len() {
            let char_start = inside[i..].chars().next().unwrap();
            let char_len = char_start.len_utf8();

            let is_whitespace = char_start.is_whitespace();
            let is_word_char = char_start.is_alphanumeric() || char_start == '_';
            let is_symbol = !is_whitespace && !is_word_char;

            match (current_type, is_whitespace, is_word_char, is_symbol) {
                (None, true, _, _) => {
                    // Continue in whitespace
                    i += char_len;
                }
                (None, false, true, _) => {
                    // Transition from whitespace to word
                    current_start = i;
                    current_end = i + char_len;
                    current_type = Some(true);
                    i += char_len;
                }
                (None, false, false, true) => {
                    // Transition from whitespace to symbol
                    current_start = i;
                    current_end = i + char_len;
                    current_type = Some(false);
                    i += char_len;
                }
                (None, false, false, false) => {
                    // This shouldn't happen since a char is either whitespace, word, or symbol
                    // But handle it gracefully by treating as symbol
                    current_start = i;
                    current_end = i + char_len;
                    current_type = Some(false);
                    i += char_len;
                }
                (Some(true), true, _, _) => {
                    // Transition from word to whitespace - finish word
                    matches.push(((current_start, current_end), true));
                    current_type = None;
                    i += char_len;
                }
                (Some(true), false, true, _) => {
                    // Continue in word
                    current_end = i + char_len;
                    i += char_len;
                }
                (Some(true), false, false, true) => {
                    // Transition from word to symbol - finish word, start symbol
                    matches.push(((current_start, current_end), true));
                    current_start = i;
                    current_end = i + char_len;
                    current_type = Some(false);
                    i += char_len;
                }
                (Some(true), false, false, false) => {
                    // This shouldn't happen, but handle as symbol
                    matches.push(((current_start, current_end), true));
                    current_start = i;
                    current_end = i + char_len;
                    current_type = Some(false);
                    i += char_len;
                }
                (Some(false), true, _, _) => {
                    // Transition from symbol to whitespace - finish symbol
                    matches.push(((current_start, current_end), true));
                    current_type = None;
                    i += char_len;
                }
                (Some(false), false, true, _) => {
                    // Transition from symbol to word - finish symbol, start word
                    matches.push(((current_start, current_end), true));
                    current_start = i;
                    current_end = i + char_len;
                    current_type = Some(true);
                    i += char_len;
                }
                (Some(false), false, false, true) => {
                    // Continue in symbol
                    current_end = i + char_len;
                    i += char_len;
                }
                (Some(false), false, false, false) => {
                    // This shouldn't happen, but handle as symbol
                    current_end = i + char_len;
                    i += char_len;
                }
            }
        }

        // Don't forget the last token
        if let Some(_) = current_type {
            matches.push(((current_start, current_end), true));
        }

        Ok(matches)
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
    fn optimized_compatibility() {
        // Test that the optimized version produces the same results as the original
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
            let mut optimized = PreTokenizedString::from(test_case);

            let original_pretok = Whitespace {};
            let optimized_pretok = WhitespaceOptimized {};

            original_pretok.pre_tokenize(&mut original).unwrap();
            optimized_pretok.pre_tokenize(&mut optimized).unwrap();

            let original_splits = original
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>();

            let optimized_splits = optimized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>();

            assert_eq!(
                original_splits, optimized_splits,
                "Mismatch for test case: '{}'",
                test_case
            );
        }
    }

    #[test]
    fn optimized_edge_cases() {
        let pretok = WhitespaceOptimized {};

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
}
