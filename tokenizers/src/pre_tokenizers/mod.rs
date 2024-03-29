pub mod bert;
pub mod byte_level;
pub mod delimiter;
pub mod digits;
pub mod metaspace;
pub mod punctuation;
pub mod sequence;
pub mod split;
pub mod unicode_scripts;
pub mod whitespace;

use serde::{Deserialize, Serialize};

use crate::pre_tokenizers::bert::BertPreTokenizer;
use crate::pre_tokenizers::byte_level::ByteLevel;
use crate::pre_tokenizers::delimiter::CharDelimiterSplit;
use crate::pre_tokenizers::digits::Digits;
use crate::pre_tokenizers::metaspace::Metaspace;
use crate::pre_tokenizers::punctuation::Punctuation;
use crate::pre_tokenizers::sequence::Sequence;
use crate::pre_tokenizers::split::Split;
use crate::pre_tokenizers::unicode_scripts::UnicodeScripts;
use crate::pre_tokenizers::whitespace::{Whitespace, WhitespaceSplit};
use crate::{PreTokenizedString, PreTokenizer};

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
#[serde(untagged)]
pub enum PreTokenizerWrapper {
    BertPreTokenizer(BertPreTokenizer),
    ByteLevel(ByteLevel),
    Delimiter(CharDelimiterSplit),
    Metaspace(Metaspace),
    Whitespace(Whitespace),
    Sequence(Sequence),
    Split(Split),
    Punctuation(Punctuation),
    WhitespaceSplit(WhitespaceSplit),
    Digits(Digits),
    UnicodeScripts(UnicodeScripts),
}

impl PreTokenizer for PreTokenizerWrapper {
    fn pre_tokenize(&self, normalized: &mut PreTokenizedString) -> crate::Result<()> {
        match self {
            Self::BertPreTokenizer(bpt) => bpt.pre_tokenize(normalized),
            Self::ByteLevel(bpt) => bpt.pre_tokenize(normalized),
            Self::Delimiter(dpt) => dpt.pre_tokenize(normalized),
            Self::Metaspace(mspt) => mspt.pre_tokenize(normalized),
            Self::Whitespace(wspt) => wspt.pre_tokenize(normalized),
            Self::Punctuation(tok) => tok.pre_tokenize(normalized),
            Self::Sequence(tok) => tok.pre_tokenize(normalized),
            Self::Split(tok) => tok.pre_tokenize(normalized),
            Self::WhitespaceSplit(wspt) => wspt.pre_tokenize(normalized),
            Self::Digits(wspt) => wspt.pre_tokenize(normalized),
            Self::UnicodeScripts(us) => us.pre_tokenize(normalized),
        }
    }
}

impl_enum_from!(BertPreTokenizer, PreTokenizerWrapper, BertPreTokenizer);
impl_enum_from!(ByteLevel, PreTokenizerWrapper, ByteLevel);
impl_enum_from!(CharDelimiterSplit, PreTokenizerWrapper, Delimiter);
impl_enum_from!(Whitespace, PreTokenizerWrapper, Whitespace);
impl_enum_from!(Punctuation, PreTokenizerWrapper, Punctuation);
impl_enum_from!(Sequence, PreTokenizerWrapper, Sequence);
impl_enum_from!(Split, PreTokenizerWrapper, Split);
impl_enum_from!(Metaspace, PreTokenizerWrapper, Metaspace);
impl_enum_from!(WhitespaceSplit, PreTokenizerWrapper, WhitespaceSplit);
impl_enum_from!(Digits, PreTokenizerWrapper, Digits);
impl_enum_from!(UnicodeScripts, PreTokenizerWrapper, UnicodeScripts);

#[cfg(test)]
mod tests {
    use super::metaspace::PrependScheme;
    use super::*;

    #[test]
    fn test_deserialize() {
        let pre_tokenizer: PreTokenizerWrapper = serde_json::from_str(r#"{"type":"Sequence","pretokenizers":[{"type":"WhitespaceSplit"},{"type":"Metaspace","replacement":"▁","str_rep":"▁","add_prefix_space":true}]}"#).unwrap();

        assert_eq!(
            pre_tokenizer,
            PreTokenizerWrapper::Sequence(Sequence::new(vec![
                PreTokenizerWrapper::WhitespaceSplit(WhitespaceSplit {}),
                PreTokenizerWrapper::Metaspace(Metaspace::new('▁', PrependScheme::Always, true))
            ]))
        );

        let pre_tokenizer: PreTokenizerWrapper = serde_json::from_str(
            r#"{"type":"Metaspace","replacement":"▁","add_prefix_space":true}"#,
        )
        .unwrap();

        assert_eq!(
            pre_tokenizer,
            PreTokenizerWrapper::Metaspace(Metaspace::new('▁', PrependScheme::Always, true))
        );

        let pre_tokenizer: PreTokenizerWrapper = serde_json::from_str(r#"{"type":"Sequence","pretokenizers":[{"type":"WhitespaceSplit"},{"type":"Metaspace","replacement":"▁","add_prefix_space":true}]}"#).unwrap();

        assert_eq!(
            pre_tokenizer,
            PreTokenizerWrapper::Sequence(Sequence::new(vec![
                PreTokenizerWrapper::WhitespaceSplit(WhitespaceSplit {}),
                PreTokenizerWrapper::Metaspace(Metaspace::new('▁', PrependScheme::Always, true))
            ]))
        );

        let pre_tokenizer: PreTokenizerWrapper = serde_json::from_str(
            r#"{"type":"Metaspace","replacement":"▁","add_prefix_space":true, "prepend_scheme":"first"}"#,
        )
        .unwrap();

        assert_eq!(
            pre_tokenizer,
            PreTokenizerWrapper::Metaspace(Metaspace::new(
                '▁',
                metaspace::PrependScheme::First,
                true
            ))
        );

        let pre_tokenizer: PreTokenizerWrapper = serde_json::from_str(
            r#"{"type":"Metaspace","replacement":"▁","add_prefix_space":true, "prepend_scheme":"always"}"#,
        )
        .unwrap();

        assert_eq!(
            pre_tokenizer,
            PreTokenizerWrapper::Metaspace(Metaspace::new(
                '▁',
                metaspace::PrependScheme::Always,
                true
            ))
        );
    }

    #[test]
    fn test_deserialize_whitespace_split() {
        let pre_tokenizer: PreTokenizerWrapper =
            serde_json::from_str(r#"{"type":"WhitespaceSplit"}"#).unwrap();
        assert_eq!(
            pre_tokenizer,
            PreTokenizerWrapper::WhitespaceSplit(WhitespaceSplit {})
        );
    }
}
