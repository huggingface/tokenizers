use crate::tokenizer::{Encoding, PostProcessor, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::iter::FromIterator;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
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
        BertProcessing { sep, cls }
    }
}

impl PostProcessor for BertProcessing {
    fn added_tokens(&self, is_pair: bool) -> usize {
        if is_pair {
            3
        } else {
            2
        }
    }

    fn process(
        &self,
        mut encoding: Encoding,
        pair_encoding: Option<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        if !add_special_tokens {
            return PostProcessor::default_process(encoding, pair_encoding, add_special_tokens);
        }

        let ids = [&[self.cls.1], &encoding.get_ids()[..], &[self.sep.1]].concat();
        let type_ids = [&[0], &encoding.get_type_ids()[..], &[0]].concat();
        let tokens = [
            &[self.cls.0.clone()],
            &encoding.get_tokens()[..],
            &[self.sep.0.clone()],
        ]
        .concat();
        let words = [&[None], &encoding.get_word_ids()[..], &[None]].concat();
        let offsets = [&[(0, 0)], &encoding.get_offsets()[..], &[(0, 0)]].concat();
        let special_tokens = [&[1u32], &vec![0; encoding.get_ids().len()][..], &[1]].concat();
        let attention_mask = vec![1; ids.len()];

        // For compatibility with `TemplateProcessing`, the sequence_ranges shouldn't contain
        // the special tokens.
        let sequence_ranges = HashMap::from_iter(vec![(0, 1..ids.len() - 1)]);
        let mut new_encoding = Encoding::new(
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
                    let ids = [&[self.cls.1], &encoding.get_ids()[..], &[self.sep.1]].concat();
                    let type_ids = [&[0], &encoding.get_type_ids()[..], &[0]].concat();
                    let tokens = [
                        &[self.cls.0.clone()],
                        &encoding.get_tokens()[..],
                        &[self.sep.0.clone()],
                    ]
                    .concat();
                    let words = [&[None], &encoding.get_word_ids()[..], &[None]].concat();
                    let offsets = [&[(0, 0)], &encoding.get_offsets()[..], &[(0, 0)]].concat();
                    let special_tokens =
                        [&[1u32], &vec![0; encoding.get_ids().len()][..], &[1]].concat();
                    let attention_mask = vec![1; ids.len()];

                    // For compatibility with `TemplateProcessing`, the sequence_ranges shouldn't
                    // contain the special tokens.
                    let sequence_ranges = HashMap::from_iter(vec![(0, 1..ids.len() - 1)]);
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
        );

        if let Some(mut encoding) = pair_encoding {
            let pair_ids = [&encoding.get_ids()[..], &[self.sep.1]].concat();
            let pair_type_ids = [&encoding.get_type_ids()[..], &[1]].concat();
            let pair_tokens = [&encoding.get_tokens()[..], &[self.sep.0.clone()]].concat();
            let pair_words = [&encoding.get_word_ids()[..], &[None]].concat();
            let pair_offsets = [&encoding.get_offsets()[..], &[(0, 0)]].concat();
            let pair_special_tokens =
                [&vec![0u32; encoding.get_type_ids().len()][..], &[1]].concat();
            let pair_attention_mask = vec![1; pair_ids.len()];

            // For compatibility with `TemplateProcessing`, the sequence_ranges shouldn't contain
            // the special tokens.
            let pair_sequence_ranges = HashMap::from_iter(vec![(1, 0..pair_ids.len() - 1)]);
            let new_pair_encoding = Encoding::new(
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
                        let pair_ids = [&encoding.get_ids()[..], &[self.sep.1]].concat();
                        let pair_type_ids = [&encoding.get_type_ids()[..], &[1]].concat();
                        let pair_tokens =
                            [&encoding.get_tokens()[..], &[self.sep.0.clone()]].concat();
                        let pair_words = [&encoding.get_word_ids()[..], &[None]].concat();
                        let pair_offsets = [&encoding.get_offsets()[..], &[(0, 0)]].concat();
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
            );

            new_encoding.merge_with(new_pair_encoding, false);
        }

        Ok(new_encoding)
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
}
