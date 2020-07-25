pub mod bert;
pub mod byte_level;
pub mod delimiter;
pub mod metaspace;
pub mod whitespace;

use serde::{Deserialize, Serialize};

use crate::pre_tokenizers::bert::BertPreTokenizer;
use crate::pre_tokenizers::byte_level::ByteLevel;
use crate::pre_tokenizers::delimiter::CharDelimiterSplit;
use crate::pre_tokenizers::metaspace::Metaspace;
use crate::pre_tokenizers::whitespace::{Whitespace, WhitespaceSplit};
use crate::{NormalizedString, PreTokenizer};

#[derive(Serialize, Deserialize)]
pub enum PreTokenizerWrapper {
    BertPreTokenizer(BertPreTokenizer),
    ByteLevel(ByteLevel),
    Delimiter(CharDelimiterSplit),
    Metaspace(Metaspace),
    Whitespace(Whitespace),
    WhitespaceSplit(WhitespaceSplit),
}

#[typetag::serde]
impl PreTokenizer for PreTokenizerWrapper {
    fn pre_tokenize(
        &self,
        normalized: &mut NormalizedString,
    ) -> crate::Result<Vec<(String, (usize, usize))>> {
        match self {
            PreTokenizerWrapper::BertPreTokenizer(bpt) => bpt.pre_tokenize(normalized),
            PreTokenizerWrapper::ByteLevel(bpt) => bpt.pre_tokenize(normalized),
            PreTokenizerWrapper::Delimiter(dpt) => dpt.pre_tokenize(normalized),
            PreTokenizerWrapper::Metaspace(mspt) => mspt.pre_tokenize(normalized),
            PreTokenizerWrapper::Whitespace(wspt) => wspt.pre_tokenize(normalized),
            PreTokenizerWrapper::WhitespaceSplit(wspt) => wspt.pre_tokenize(normalized),
        }
    }
}

impl_enum_from!(BertPreTokenizer, PreTokenizerWrapper, BertPreTokenizer);
impl_enum_from!(ByteLevel, PreTokenizerWrapper, ByteLevel);
impl_enum_from!(CharDelimiterSplit, PreTokenizerWrapper, Delimiter);
impl_enum_from!(Whitespace, PreTokenizerWrapper, Whitespace);
impl_enum_from!(Metaspace, PreTokenizerWrapper, Metaspace);
impl_enum_from!(WhitespaceSplit, PreTokenizerWrapper, WhitespaceSplit);
