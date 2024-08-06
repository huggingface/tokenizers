pub mod bert;
pub mod roberta;
pub mod sequence;
pub mod template;

// Re-export these as processors
pub use super::pre_tokenizers::byte_level;

use serde::{Deserialize, Serialize};

use crate::pre_tokenizers::byte_level::ByteLevel;
use crate::processors::bert::BertProcessing;
use crate::processors::roberta::RobertaProcessing;
use crate::processors::sequence::Sequence;
use crate::processors::template::TemplateProcessing;
use crate::{Encoding, PostProcessor, Result};

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone, Eq)]
#[serde(untagged)]
pub enum PostProcessorWrapper {
    // Roberta must be before Bert for deserialization (serde does not validate tags)
    Roberta(RobertaProcessing),
    Bert(BertProcessing),
    ByteLevel(ByteLevel),
    Template(TemplateProcessing),
    Sequence(Sequence),
}

impl PostProcessor for PostProcessorWrapper {
    fn added_tokens(&self, is_pair: bool) -> usize {
        match self {
            Self::Bert(bert) => bert.added_tokens(is_pair),
            Self::ByteLevel(bl) => bl.added_tokens(is_pair),
            Self::Roberta(roberta) => roberta.added_tokens(is_pair),
            Self::Template(template) => template.added_tokens(is_pair),
            Self::Sequence(bl) => bl.added_tokens(is_pair),
        }
    }

    fn process_encodings(
        &self,
        encodings: Vec<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>> {
        match self {
            Self::Bert(bert) => bert.process_encodings(encodings, add_special_tokens),
            Self::ByteLevel(bl) => bl.process_encodings(encodings, add_special_tokens),
            Self::Roberta(roberta) => roberta.process_encodings(encodings, add_special_tokens),
            Self::Template(template) => template.process_encodings(encodings, add_special_tokens),
            Self::Sequence(bl) => bl.process_encodings(encodings, add_special_tokens),
        }
    }
}

impl_enum_from!(BertProcessing, PostProcessorWrapper, Bert);
impl_enum_from!(ByteLevel, PostProcessorWrapper, ByteLevel);
impl_enum_from!(RobertaProcessing, PostProcessorWrapper, Roberta);
impl_enum_from!(TemplateProcessing, PostProcessorWrapper, Template);
impl_enum_from!(Sequence, PostProcessorWrapper, Sequence);

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

    #[test]
    fn post_processor_deserialization_no_type() {
        let json = r#"{"add_prefix_space": true, "trim_offsets": false, "use_regex": false}"#;
        let reconstructed = serde_json::from_str::<PostProcessorWrapper>(json);
        match reconstructed {
            Err(err) => assert_eq!(
                err.to_string(),
                "data did not match any variant of untagged enum PostProcessorWrapper"
            ),
            _ => panic!("Expected an error here"),
        }

        let json = r#"{"sep":["[SEP]",102],"cls":["[CLS]",101]}"#;
        let reconstructed = serde_json::from_str::<PostProcessorWrapper>(json);
        assert!(matches!(
            reconstructed.unwrap(),
            PostProcessorWrapper::Bert(_)
        ));

        let json =
            r#"{"sep":["</s>",2], "cls":["<s>",0], "trim_offsets":true, "add_prefix_space":true}"#;
        let reconstructed = serde_json::from_str::<PostProcessorWrapper>(json);
        assert!(matches!(
            reconstructed.unwrap(),
            PostProcessorWrapper::Roberta(_)
        ));

        let json = r#"{"type":"RobertaProcessing", "sep":["</s>",2] }"#;
        let reconstructed = serde_json::from_str::<PostProcessorWrapper>(json);
        match reconstructed {
            Err(err) => assert_eq!(
                err.to_string(),
                "data did not match any variant of untagged enum PostProcessorWrapper"
            ),
            _ => panic!("Expected an error here"),
        }
    }
}
