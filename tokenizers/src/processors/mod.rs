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

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum PostProcessorWrapper {
    Bert(BertProcessing),
    ByteLevel(ByteLevel),
    Roberta(RobertaProcessing),
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
