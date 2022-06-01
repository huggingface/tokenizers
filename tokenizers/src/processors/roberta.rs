use crate::processors::byte_level::process_offsets;
use crate::tokenizer::{Encoding, PostProcessor, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::iter::FromIterator;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "type")]
pub struct RobertaProcessing {
    sep: (String, u32),
    cls: (String, u32),
    trim_offsets: bool,
    add_prefix_space: bool,
}

impl Default for RobertaProcessing {
    fn default() -> Self {
        Self {
            sep: ("</s>".into(), 2),
            cls: ("<s>".into(), 0),
            trim_offsets: true,
            add_prefix_space: true,
        }
    }
}

impl RobertaProcessing {
    pub fn new(sep: (String, u32), cls: (String, u32)) -> Self {
        Self {
            sep,
            cls,
            ..Default::default()
        }
    }

    #[must_use]
    pub fn trim_offsets(mut self, v: bool) -> Self {
        self.trim_offsets = v;
        self
    }

    #[must_use]
    pub fn add_prefix_space(mut self, v: bool) -> Self {
        self.add_prefix_space = v;
        self
    }
}

impl PostProcessor for RobertaProcessing {
    fn added_tokens(&self, is_pair: bool) -> usize {
        if is_pair {
            4
        } else {
            2
        }
    }

    fn process_chain(
        &self,
        encodings: Vec<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>> {
        let encodings = encodings
            .into_iter()
            .map(|mut encoding| {
                if self.trim_offsets {
                    process_offsets(&mut encoding, self.add_prefix_space);
                    encoding
                        .get_overflowing_mut()
                        .iter_mut()
                        .for_each(|encoding| process_offsets(encoding, self.add_prefix_space));
                }

                if !add_special_tokens {
                    return encoding;
                }

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
                )
            })
            .collect::<Vec<_>>();

        Ok(encodings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde() {
        let roberta = RobertaProcessing::default();
        let roberta_r = r#"{
            "type":"RobertaProcessing",
            "sep":["</s>",2],
            "cls":["<s>",0],
            "trim_offsets":true,
            "add_prefix_space":true
        }"#
        .replace(char::is_whitespace, "");
        assert_eq!(serde_json::to_string(&roberta).unwrap(), roberta_r);
        assert_eq!(
            serde_json::from_str::<RobertaProcessing>(&roberta_r).unwrap(),
            roberta
        );
    }
}
