use crate::tokenizer::{Encoding, PostProcessor, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::iter::FromIterator;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(tag = "type")]
pub struct BertProcessing {
    sep: (String, u32),
    cls: (String, u32),
}

impl Default for BertProcessing {
    fn default() -> Self {
        Self {
            sep: ("[SEP]".into(), 102),
            cls: ("[CLS]".into(), 101),
        }
    }
}

impl BertProcessing {
    pub fn new(sep: (String, u32), cls: (String, u32)) -> Self {
        Self { sep, cls }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum BertProcessorError {
    #[error("encodings vector length must be either 1 or 2")]
    InvalidEncodingsVecLength,
}

impl PostProcessor for BertProcessing {
    fn added_tokens(&self, is_pair: bool) -> usize {
        if is_pair {
            3
        } else {
            2
        }
    }

    fn process_encodings(
        &self,
        mut encodings: Vec<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>> {
        if !add_special_tokens {
            return Ok(encodings);
        }

        let encodings: Vec<Encoding> = encodings
            .iter_mut()
            .enumerate()
            .map(|(i, encoding)| {
                if i == 0 {
                    let ids = [&[self.cls.1], encoding.get_ids(), &[self.sep.1]].concat();
                    let type_ids = [&[0], encoding.get_type_ids(), &[0]].concat();
                    let tokens = [
                        &[self.cls.0.clone()],
                        encoding.get_tokens(),
                        &[self.sep.0.clone()],
                    ]
                    .concat();
                    let words = [&[None], encoding.get_word_ids(), &[None]].concat();
                    let offsets = [&[(0, 0)], encoding.get_offsets(), &[(0, 0)]].concat();
                    let special_tokens =
                        [&[1u32], &vec![0; encoding.get_ids().len()][..], &[1]].concat();
                    let attention_mask = vec![1; ids.len()];

                    // For compatibility with `TemplateProcessing`, the sequence_ranges shouldn't contain
                    // the special tokens.
                    let sequence_ranges = HashMap::from_iter(vec![(0, 1..ids.len() - 1)]);
                    Encoding::new(
                        ids,
                        type_ids,
                        tokens,
                        words,
                        offsets,
                        special_tokens,
                        attention_mask,
                        encoding
                            .take_overflowing()
                            .into_iter()
                            .map(|encoding| {
                                let ids =
                                    [&[self.cls.1], encoding.get_ids(), &[self.sep.1]].concat();
                                let type_ids = [&[0], encoding.get_type_ids(), &[0]].concat();
                                let tokens = [
                                    &[self.cls.0.clone()],
                                    encoding.get_tokens(),
                                    &[self.sep.0.clone()],
                                ]
                                .concat();
                                let words = [&[None], encoding.get_word_ids(), &[None]].concat();
                                let offsets =
                                    [&[(0, 0)], encoding.get_offsets(), &[(0, 0)]].concat();
                                let special_tokens =
                                    [&[1u32], &vec![0; encoding.get_ids().len()][..], &[1]]
                                        .concat();
                                let attention_mask = vec![1; ids.len()];

                                // For compatibility with `TemplateProcessing`, the sequence_ranges shouldn't
                                // contain the special tokens.
                                let sequence_ranges =
                                    HashMap::from_iter(vec![(0, 1..ids.len() - 1)]);
                                Encoding::new(
                                    ids,
                                    type_ids,
                                    tokens,
                                    words,
                                    offsets,
                                    special_tokens,
                                    attention_mask,
                                    vec![],
                                    sequence_ranges,
                                )
                            })
                            .collect(),
                        sequence_ranges,
                    )
                } else {
                    let pair_ids = [encoding.get_ids(), &[self.sep.1]].concat();
                    let pair_type_ids = [encoding.get_type_ids(), &[1]].concat();
                    let pair_tokens = [encoding.get_tokens(), &[self.sep.0.clone()]].concat();
                    let pair_words = [encoding.get_word_ids(), &[None]].concat();
                    let pair_offsets = [encoding.get_offsets(), &[(0, 0)]].concat();
                    let pair_special_tokens =
                        [&vec![0u32; encoding.get_type_ids().len()][..], &[1]].concat();
                    let pair_attention_mask = vec![1; pair_ids.len()];

                    // For compatibility with `TemplateProcessing`, the sequence_ranges shouldn't contain
                    // the special tokens.
                    let pair_sequence_ranges = HashMap::from_iter(vec![(1, 0..pair_ids.len() - 1)]);
                    Encoding::new(
                        pair_ids,
                        pair_type_ids,
                        pair_tokens,
                        pair_words,
                        pair_offsets,
                        pair_special_tokens,
                        pair_attention_mask,
                        encoding
                            .take_overflowing()
                            .into_iter()
                            .map(|encoding| {
                                let pair_ids = [encoding.get_ids(), &[self.sep.1]].concat();
                                let pair_type_ids = [encoding.get_type_ids(), &[1]].concat();
                                let pair_tokens =
                                    [encoding.get_tokens(), &[self.sep.0.clone()]].concat();
                                let pair_words = [encoding.get_word_ids(), &[None]].concat();
                                let pair_offsets = [encoding.get_offsets(), &[(0, 0)]].concat();
                                let pair_special_tokens =
                                    [&vec![0u32; encoding.get_type_ids().len()][..], &[1]].concat();
                                let pair_attention_mask = vec![1; pair_ids.len()];

                                // For compatibility with `TemplateProcessing`, the sequence_ranges
                                // shouldn't contain the special tokens.
                                let pair_sequence_ranges =
                                    HashMap::from_iter(vec![(1, 0..pair_ids.len() - 1)]);
                                Encoding::new(
                                    pair_ids,
                                    pair_type_ids,
                                    pair_tokens,
                                    pair_words,
                                    pair_offsets,
                                    pair_special_tokens,
                                    pair_attention_mask,
                                    vec![],
                                    pair_sequence_ranges,
                                )
                            })
                            .collect(),
                        pair_sequence_ranges,
                    )
                }
            })
            .collect();

        Ok(encodings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde() {
        let bert = BertProcessing::default();
        let bert_r = r#"{"type":"BertProcessing","sep":["[SEP]",102],"cls":["[CLS]",101]}"#;
        assert_eq!(serde_json::to_string(&bert).unwrap(), bert_r);
        assert_eq!(
            serde_json::from_str::<BertProcessing>(bert_r).unwrap(),
            bert
        );
    }

    #[test]
    fn bert_processing() {
        let processor = BertProcessing::default();
        assert_eq!(processor.added_tokens(false), 2);
        assert_eq!(processor.added_tokens(true), 3);

        use crate::Token;
        let encoding = Encoding::from_tokens(
            vec![
                Token::new(12, "Hello".into(), (0, 5)),
                Token::new(14, "there".into(), (6, 11)),
            ],
            0,
        );
        let pair = Encoding::from_tokens(vec![Token::new(15, "pair".into(), (0, 4))], 0);
        let single_encoding = processor.process(encoding.clone(), None, true).unwrap();
        assert_eq!(
            single_encoding,
            Encoding::new(
                vec![101, 12, 14, 102],
                vec![0, 0, 0, 0],
                vec![
                    "[CLS]".into(),
                    "Hello".into(),
                    "there".into(),
                    "[SEP]".into()
                ],
                vec![None, None, None, None],
                vec![(0, 0), (0, 5), (6, 11), (0, 0)],
                vec![1, 0, 0, 1],
                vec![1, 1, 1, 1],
                vec![],
                HashMap::from_iter(vec![(0, 1..3)]),
            )
        );
        assert_eq!(single_encoding.token_to_sequence(2), Some(0));
        assert_eq!(single_encoding.token_to_sequence(3), None);
        let pair_encoding = processor
            .process(encoding.clone(), Some(pair.clone()), true)
            .unwrap();
        assert_eq!(
            pair_encoding,
            Encoding::new(
                vec![101, 12, 14, 102, 15, 102],
                vec![0, 0, 0, 0, 1, 1],
                vec![
                    "[CLS]".into(),
                    "Hello".into(),
                    "there".into(),
                    "[SEP]".into(),
                    "pair".into(),
                    "[SEP]".into()
                ],
                vec![None, None, None, None, None, None],
                vec![(0, 0), (0, 5), (6, 11), (0, 0), (0, 4), (0, 0)],
                vec![1, 0, 0, 1, 0, 1],
                vec![1, 1, 1, 1, 1, 1],
                vec![],
                HashMap::from_iter(vec![(0, 1..3), (1, 4..5)]),
            )
        );
        assert_eq!(pair_encoding.token_to_sequence(2), Some(0));
        assert_eq!(pair_encoding.token_to_sequence(3), None);
        assert_eq!(pair_encoding.token_to_sequence(4), Some(1));
        assert_eq!(pair_encoding.token_to_sequence(5), None);

        // No special tokens
        let pair_encoding = processor.process(encoding, Some(pair), false).unwrap();
        assert_eq!(
            pair_encoding,
            Encoding::new(
                vec![12, 14, 15],
                vec![0, 0, 1],
                vec!["Hello".into(), "there".into(), "pair".into(),],
                vec![None, None, None],
                vec![(0, 5), (6, 11), (0, 4)],
                vec![0, 0, 0],
                vec![1, 1, 1],
                vec![],
                HashMap::from_iter(vec![(0, 0..2), (1, 2..3)]),
            )
        );
        assert_eq!(pair_encoding.token_to_sequence(0), Some(0));
        assert_eq!(pair_encoding.token_to_sequence(1), Some(0));
        assert_eq!(pair_encoding.token_to_sequence(2), Some(1));
    }
}
