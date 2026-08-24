use crate::pipeline::{self, PreTokenizerScratch};
use crate::tokenizer::Result;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Whitespace;

use bitsplit::classify::mask;

impl Default for Whitespace {
    fn default() -> Self {
        Self
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct WhitespaceSplit;

// SAFETY: the spans come from an `bitsplit` fsm, which cuts only at character boundaries of `text`.
// See "What the spans guarantee" in the `bitsplit` docs.
unsafe impl pipeline::PreTokenizer for WhitespaceSplit {
    fn pre_tokenize(
        &self,
        text: &str,
        scratch: &mut PreTokenizerScratch,
        out: &mut Vec<pipeline::Span>,
    ) -> Result<()> {
        // drop whitespace runs, keep everything else as runs — bitsplit SIMD classify + class-runs FSM.
        // atom `WS` == `char::is_whitespace`, so byte-exact with the scalar path.
        scratch.split_on_bits(
            text.as_bytes(),
            |b, t, st, fk, _, o| {
                bitsplit::classes::class_runs_into::<{ mask::WS }, 0, 0>(b, t, st, fk, o)
            },
            out,
        );
        Ok(())
    }
}

/// Matches the same characters as the `\w` regex class (Unicode-aware).
/// This is: Alphabetic + Nd (decimal digit) + Pc (connector punctuation) +
/// M (marks) + Join_Control — NOT Nl/No (which Rust's is_alphanumeric includes).
pub fn is_word_char(ch: char) -> bool {
    use unicode_categories::UnicodeCategories;

    ch.is_alphabetic()
        || ch.is_number_decimal_digit()
        || ch.is_punctuation_connector()
        || ch.is_mark()
        || ch == '\u{200c}' // Zero-Width Non-Joiner
        || ch == '\u{200d}' // Zero-Width Joiner
}

// SAFETY: the spans come from an `bitsplit` fsm, which splits only at character boundaries of `text`.
// See `bitsplit` docs.
unsafe impl pipeline::PreTokenizer for Whitespace {
    #[inline(never)]
    fn pre_tokenize(
        &self,
        text: &str,
        scratch: &mut PreTokenizerScratch,
        out: &mut Vec<pipeline::Span>,
    ) -> Result<()> {
        // `\w+|[^\w\s]+`: drop whitespace, cut at the word↔symbol boundary, each run one token —
        // bitsplit classify + class-runs FSM (`WORD` = `\w`; keep-A = word, keep-B = symbol).
        scratch.split_on_bits(
            text.as_bytes(),
            |b, t, st, fk, _, o| {
                bitsplit::classes::class_runs_into::<{ mask::WS }, 0, { mask::WORD }>(
                    b, t, st, fk, o,
                )
            },
            out,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pretokenize(text: &str) -> Vec<(&str, (u32, u32))> {
        let pretok = Whitespace;
        let mut scratch = PreTokenizerScratch::default();
        let mut splits = Vec::new();
        crate::pipeline::PreTokenizer::pre_tokenize(&pretok, text, &mut scratch, &mut splits)
            .unwrap();
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

    fn pretokenize_split(text: &str) -> Vec<(&str, (u32, u32))> {
        let pretok = WhitespaceSplit;
        let mut scratch = PreTokenizerScratch::default();
        let mut splits = Vec::new();
        crate::pipeline::PreTokenizer::pre_tokenize(&pretok, text, &mut scratch, &mut splits)
            .unwrap();
        splits
            .iter()
            .map(|s| (&text[s.range()], (s.start, s.end)))
            .collect()
    }

    #[test]
    fn whitespace_split_pipeline() {
        // same cases as `whitespace_split`, via the pipeline path: only whitespace
        // splits, punctuation stays attached to its run
        assert_eq!(
            pretokenize_split("Hey man!"),
            vec![("Hey", (0, 3)), ("man!", (4, 8))],
        );
        assert_eq!(
            pretokenize_split("Hey, man, Good?"),
            vec![("Hey,", (0, 4)), ("man,", (5, 9)), ("Good?", (10, 15))],
        );
    }

    #[test]
    fn whitespace_split_pipeline_edge_cases() {
        let empty = Vec::<(&str, (u32, u32))>::new();
        assert_eq!(pretokenize_split(""), empty);
        assert_eq!(pretokenize_split("   "), empty);
        assert_eq!(pretokenize_split("a"), vec![("a", (0, 1))]);
        // leading/trailing/multiple whitespace dropped, runs kept whole
        assert_eq!(
            pretokenize_split(" a  b "),
            vec![("a", (1, 2)), ("b", (4, 5))]
        );
        assert_eq!(
            pretokenize_split("a\tb\nc"),
            vec![("a", (0, 1)), ("b", (2, 3)), ("c", (4, 5))],
        );
        // multibyte: é is 2 bytes -> byte offsets
        assert_eq!(
            pretokenize_split("café au lait"),
            vec![("café", (0, 5)), ("au", (6, 8)), ("lait", (9, 13))],
        );
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
