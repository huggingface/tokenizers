//! `PostProcessorWrapper`: the `post_processor` field of a `tokenizer.json`.
//!
//! Untagged, and the variant order is load-bearing: serde does not validate a `"type"` tag against
//! an untagged enum, so `Roberta` must stay before `Bert` (a Roberta config satisfies Bert's shape).

pub mod mirror;
pub mod sequence;

pub use sequence::Sequence as SequenceProcessor;

use serde::{Deserialize, Serialize};

use tk_encode::pre_tokenizers::byte_level::ByteLevel;
use tk_encode::processors::bert::BertProcessing;
use tk_encode::processors::roberta::RobertaProcessing;
use tk_encode::processors::template::TemplateProcessing;
use tk_encode::{Encoding, PostProcessor, Result};

use crate::macros::impl_enum_from;
use crate::processors::sequence::Sequence;

/// Three of the five variants now name a mirror in [`mirror`]: `tk-encode` defines the processor
/// types and derives serde on none of them, so the on-disk shape of each one is described there.
///
/// The two that do not: `Sequence` is this crate's own type, so the orphan rule never applied to it;
/// `ByteLevel` is a *pre-tokenizer* that doubles as a post-processor, and its serde still lives in
/// `tk-encode` because it belongs to the `pre_tokenizers` half of the migration — one type, one
/// mirror, wherever that mirror ends up.
#[derive(PartialEq, Debug, Clone, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PostProcessorWrapper {
    // Roberta must be before Bert for deserialization (serde does not validate tags)
    #[serde(with = "mirror::RobertaProcessingDef")]
    Roberta(RobertaProcessing),
    #[serde(with = "mirror::BertProcessingDef")]
    Bert(BertProcessing),
    // The same type as the `ByteLevel` *pre-tokenizer* (`tk_encode::processors` re-exports
    // `pre_tokenizers::byte_level`), so it shares that one's mirror rather than growing a second
    // description of the same three fields.
    #[serde(with = "crate::pre_tokenizers::mirror::byte_level")]
    ByteLevel(ByteLevel),
    #[serde(with = "mirror::template")]
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

// Every test in here round-trips a wrapper through serde, so the module goes with `config`.
#[cfg(test)]
mod tests {
    use super::*;

    /// `BertProcessing` and `RobertaProcessing` carry no serde themselves — `tk-encode` links none,
    /// and their shapes are owned by `mirror::{BertProcessingDef, RobertaProcessingDef}`. So the
    /// wrapper is the unit under test in both directions; being untagged, it serializes to exactly
    /// what the leaf used to serialize to.
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
        assert_eq!(
            serde_json::to_string(&PostProcessorWrapper::Roberta(roberta.clone())).unwrap(),
            roberta_r
        );
        assert_eq!(
            serde_json::from_str::<PostProcessorWrapper>(&roberta_r).unwrap(),
            PostProcessorWrapper::Roberta(roberta)
        );

        let bert = BertProcessing::default();
        let bert_r = r#"{"type":"BertProcessing","sep":["[SEP]",102],"cls":["[CLS]",101]}"#;
        assert_eq!(
            serde_json::to_string(&PostProcessorWrapper::Bert(bert.clone())).unwrap(),
            bert_r
        );
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
