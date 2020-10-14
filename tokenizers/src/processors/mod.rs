pub mod bert;
pub mod roberta;
pub mod template;

// Re-export these as processors
pub use super::pre_tokenizers::byte_level;

use serde::{Deserialize, Serialize};

use crate::pre_tokenizers::byte_level::ByteLevel;
use crate::processors::bert::BertProcessing;
use crate::processors::roberta::RobertaProcessing;
use crate::processors::template::TemplateProcessing;
use crate::{Encoding, PostProcessor, Result};

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
#[serde(untagged)]
pub enum PostProcessorWrapper {
    // Roberta must be before Bert for deserialization (serde does not validate tags)
    Roberta(RobertaProcessing),
    Bert(BertProcessing),
    ByteLevel(ByteLevel),
    Template(TemplateProcessing),
}

impl PostProcessor for PostProcessorWrapper {
    fn added_tokens(&self, is_pair: bool) -> usize {
        match self {
            PostProcessorWrapper::Bert(bert) => bert.added_tokens(is_pair),
            PostProcessorWrapper::ByteLevel(bl) => bl.added_tokens(is_pair),
            PostProcessorWrapper::Roberta(roberta) => roberta.added_tokens(is_pair),
            PostProcessorWrapper::Template(template) => template.added_tokens(is_pair),
        }
    }

    fn process(
        &self,
        encoding: Encoding,
        pair_encoding: Option<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        match self {
            PostProcessorWrapper::Bert(bert) => {
                bert.process(encoding, pair_encoding, add_special_tokens)
            }
            PostProcessorWrapper::ByteLevel(bl) => {
                bl.process(encoding, pair_encoding, add_special_tokens)
            }
            PostProcessorWrapper::Roberta(roberta) => {
                roberta.process(encoding, pair_encoding, add_special_tokens)
            }
            PostProcessorWrapper::Template(template) => {
                template.process(encoding, pair_encoding, add_special_tokens)
            }
        }
    }
}

impl_enum_from!(BertProcessing, PostProcessorWrapper, Bert);
impl_enum_from!(ByteLevel, PostProcessorWrapper, ByteLevel);
impl_enum_from!(RobertaProcessing, PostProcessorWrapper, Roberta);
impl_enum_from!(TemplateProcessing, PostProcessorWrapper, Template);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_bert_roberta_correctly() {
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
            serde_json::from_str::<PostProcessorWrapper>(&roberta_r).unwrap(),
            PostProcessorWrapper::Roberta(roberta)
        );

        let bert = BertProcessing::default();
        let bert_r = r#"{"type":"BertProcessing","sep":["[SEP]",102],"cls":["[CLS]",101]}"#;
        assert_eq!(serde_json::to_string(&bert).unwrap(), bert_r);
        assert_eq!(
            serde_json::from_str::<PostProcessorWrapper>(bert_r).unwrap(),
            PostProcessorWrapper::Bert(bert)
        );
    }
}
