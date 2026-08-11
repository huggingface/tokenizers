use crate::tokenizer::PostProcessor;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(tag = "type")]
pub struct BertProcessing {
    pub sep: (String, u32),
    pub cls: (String, u32),
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

    pub fn get_sep_copy(&self) -> (String, u32) {
        (self.sep.0.clone(), self.sep.1)
    }

    pub fn get_cls_copy(&self) -> (String, u32) {
        (self.cls.0.clone(), self.cls.1)
    }
}

impl PostProcessor for BertProcessing {}

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
