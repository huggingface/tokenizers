use crate::tokenizer::PostProcessor;
use serde::{Deserialize, Serialize};

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
    fn counts_added_tokens() {
        let processor = RobertaProcessing::default();
        assert_eq!(processor.added_tokens(false), 2);
        assert_eq!(processor.added_tokens(true), 4);
    }
}
