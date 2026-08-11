use crate::processors::byte_level::process_offsets;
use crate::tokenizer::{Encoding, PostProcessor, Result};
use ahash::AHashMap;
use serde::{Deserialize, Serialize};
use std::iter::FromIterator;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "type")]
pub struct RobertaProcessing {
    pub sep: (String, u32),
    pub cls: (String, u32),
    pub trim_offsets: bool,
    pub add_prefix_space: bool,
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

    pub fn get_sep_copy(&self) -> (String, u32) {
        (self.sep.0.clone(), self.sep.1)
    }

    pub fn get_cls_copy(&self) -> (String, u32) {
        (self.cls.0.clone(), self.cls.1)
    }
}

impl PostProcessor for RobertaProcessing {
    fn added_tokens(&self, is_pair: bool) -> usize {
        if is_pair { 4 } else { 2 }
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

    #[test]
    fn roberta_processing() {
        let processor = RobertaProcessing::default();
        assert_eq!(processor.added_tokens(false), 2);
        assert_eq!(processor.added_tokens(true), 4);

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
                vec![0, 12, 14, 2],
                vec![0, 0, 0, 0],
                vec!["<s>".into(), "Hello".into(), "there".into(), "</s>".into()],
                vec![None, None, None, None],
                vec![(0, 0), (0, 5), (6, 11), (0, 0)],
                vec![1, 0, 0, 1],
                vec![1, 1, 1, 1],
                vec![],
                AHashMap::from_iter(vec![(0, 1..3)]),
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
                vec![0, 12, 14, 2, 2, 15, 2],
                vec![0, 0, 0, 0, 0, 0, 0],
                vec![
                    "<s>".into(),
                    "Hello".into(),
                    "there".into(),
                    "</s>".into(),
                    "</s>".into(),
                    "pair".into(),
                    "</s>".into()
                ],
                vec![None, None, None, None, None, None, None],
                vec![(0, 0), (0, 5), (6, 11), (0, 0), (0, 0), (0, 4), (0, 0)],
                vec![1, 0, 0, 1, 1, 0, 1],
                vec![1, 1, 1, 1, 1, 1, 1],
                vec![],
                AHashMap::from_iter(vec![(0, 1..3), (1, 5..6)]),
            )
        );
        assert_eq!(pair_encoding.token_to_sequence(2), Some(0));
        assert_eq!(pair_encoding.token_to_sequence(3), None);
        assert_eq!(pair_encoding.token_to_sequence(4), None);
        assert_eq!(pair_encoding.token_to_sequence(5), Some(1));
        assert_eq!(pair_encoding.token_to_sequence(6), None);

        // No special tokens
        let pair_encoding = processor.process(encoding, Some(pair), false).unwrap();
        assert_eq!(
            pair_encoding,
            Encoding::new(
                vec![12, 14, 15],
                vec![0, 0, 0],
                vec!["Hello".into(), "there".into(), "pair".into(),],
                vec![None, None, None],
                vec![(0, 5), (6, 11), (0, 4)],
                vec![0, 0, 0],
                vec![1, 1, 1],
                vec![],
                AHashMap::from_iter(vec![(0, 0..2), (1, 2..3)]),
            )
        );
        assert_eq!(pair_encoding.token_to_sequence(0), Some(0));
        assert_eq!(pair_encoding.token_to_sequence(1), Some(0));
        assert_eq!(pair_encoding.token_to_sequence(2), Some(1));
    }
}
