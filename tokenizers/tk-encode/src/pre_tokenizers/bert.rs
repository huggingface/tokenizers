use crate::pipeline::{self, PreTokenizerScratch};
use crate::tokenizer::Result;

use atomsplit::classify::mask;
use atomsplit::fsm::class_runs_into;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BertPreTokenizer;

// SAFETY: the spans come from an `atomsplit` fsm, which splits only at character boundaries of `text`.
// See `atomsplit::fsm` docs.
unsafe impl pipeline::PreTokenizer for BertPreTokenizer {
    #[inline(never)]
    fn pre_tokenize(
        &self,
        text: &str,
        scratch: &mut PreTokenizerScratch,
        out: &mut Vec<pipeline::Span>,
    ) -> Result<()> {
        // Bert pre-tokenization = drop whitespace runs, isolate each punctuation char, keep every other
        // run. One `atomsplit` SIMD classify (bytes → atom tags) + the class-runs FSM, byte-exact with
        // the legacy `char::is_whitespace` / `is_punc` split above (see the tests).
        scratch.split_on_tags(
            text.as_bytes(),
            class_runs_into::<{ mask::WS }, { mask::PUNCT }, 0>,
            out,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::BertPreTokenizer;

    use crate::pipeline::PreTokenizerScratch;
    fn pretokenize(text: &str) -> Vec<(&str, (u32, u32))> {
        let pretok = BertPreTokenizer;
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
    fn basic_new() {
        assert_eq!(
            pretokenize("   Hey friend!     How are you?!?  "),
            vec![
                ("Hey", (3, 6)),
                ("friend", (7, 13)),
                ("!", (13, 14)),
                ("How", (19, 22)),
                ("are", (23, 26)),
                ("you", (27, 30)),
                ("?", (30, 31)),
                ("!", (31, 32)),
                ("?", (32, 33)),
            ],
        );
    }

    #[test]
    fn edge_cases() {
        #[allow(clippy::type_complexity)]
        let cases: Vec<(&str, Vec<(&str, (u32, u32))>)> = vec![
            ("", vec![]),
            (" ", vec![]),
            ("  ", vec![]),
            ("a", vec![("a", (0, 1))]),
            ("!", vec![("!", (0, 1))]),
            ("a!", vec![("a", (0, 1)), ("!", (1, 2))]),
            ("!a", vec![("!", (0, 1)), ("a", (1, 2))]),
            (
                "a!!b",
                vec![("a", (0, 1)), ("!", (1, 2)), ("!", (2, 3)), ("b", (3, 4))],
            ),
            ("a b", vec![("a", (0, 1)), ("b", (2, 3))]),
            ("a  b", vec![("a", (0, 1)), ("b", (3, 4))]),
            (
                "you?!?",
                vec![("you", (0, 3)), ("?", (3, 4)), ("!", (4, 5)), ("?", (5, 6))],
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(pretokenize(input), expected, "input: {input:?}");
        }
    }

    #[test]
    fn multibyte_offsets() {
        assert_eq!(
            pretokenize("café résumé"),
            vec![("café", (0, 5)), ("résumé", (6, 14))],
        );
        assert_eq!(
            pretokenize("中文 text"),
            vec![("中文", (0, 6)), ("text", (7, 11))],
        );
        assert_eq!(
            pretokenize("中 文 text"),
            vec![("中", (0, 3)), ("文", (4, 7)), ("text", (8, 12))],
        );
    }
}
