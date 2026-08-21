use crate::pipeline::{self, PreTokenizerScratch};
use crate::tokenizer::{PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior};


use super::punctuation::is_punc;
use atomsplit::classify::mask;
use atomsplit::fsm::class_runs_into;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BertPreTokenizer;

impl PreTokenizer for BertPreTokenizer {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, s| s.split(char::is_whitespace, SplitDelimiterBehavior::Removed))?;
        pretokenized.split(|_, s| s.split(is_punc, SplitDelimiterBehavior::Isolated))
    }
}

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
    use crate::{NormalizedString, OffsetReferential, OffsetType, PreTokenizedString};

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
    fn basic() {
        use crate::PreTokenizer;

        let pretok = BertPreTokenizer;
        let mut pretokenized: PreTokenizedString = "Hey friend!     How are you?!?".into();
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![
                ("Hey", (0, 3)),
                ("friend", (4, 10)),
                ("!", (10, 11)),
                ("How", (16, 19)),
                ("are", (20, 23)),
                ("you", (24, 27)),
                ("?", (27, 28)),
                ("!", (28, 29)),
                ("?", (29, 30)),
            ]
        );
    }

    #[test]
    fn chinese_chars() {
        use crate::PreTokenizer;

        let mut n = NormalizedString::from("野口里佳 Noguchi Rika");
        n.transform(
            n.get().to_owned().chars().flat_map(|c| {
                if (c as usize) > 0x4E00 {
                    vec![(' ', 0), (c, 1), (' ', 1)]
                } else {
                    vec![(c, 0)]
                }
            }),
            0,
        );
        let mut pretokenized = n.into();
        let pretok = BertPreTokenizer;
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![
                ("野", (0, 3)),
                ("口", (3, 6)),
                ("里", (6, 9)),
                ("佳", (9, 12)),
                ("Noguchi", (13, 20)),
                ("Rika", (21, 25))
            ]
        );
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
    fn chinese_chars_new() {
        let mut n = NormalizedString::from("野口里佳 Noguchi Rika");
        n.transform(
            n.get().to_owned().chars().flat_map(|c| {
                if (c as usize) > 0x4E00 {
                    vec![(' ', 0), (c, 1), (' ', 1)]
                } else {
                    vec![(c, 0)]
                }
            }),
            0,
        );
        assert_eq!(
            pretokenize(n.get()),
            vec![
                ("野", (1, 4)),
                ("口", (6, 9)),
                ("里", (11, 14)),
                ("佳", (16, 19)),
                ("Noguchi", (21, 28)),
                ("Rika", (29, 33)),
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
