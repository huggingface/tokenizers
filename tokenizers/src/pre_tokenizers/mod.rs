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

use serde::{Deserialize, Deserializer, Serialize};

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

#[derive(Serialize, Clone, Debug, PartialEq)]
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

impl<'de> Deserialize<'de> for PreTokenizerWrapper {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        pub struct Tagged {
            #[serde(rename = "type")]
            variant: EnumType,
            #[serde(flatten)]
            rest: serde_json::Value,
        }
        #[derive(Deserialize, Serialize)]
        pub enum EnumType {
            BertPreTokenizer,
            ByteLevel,
            Delimiter,
            Metaspace,
            Whitespace,
            Sequence,
            Split,
            Punctuation,
            WhitespaceSplit,
            Digits,
            UnicodeScripts,
        }

        #[derive(Deserialize)]
        #[serde(untagged)]
        pub enum PreTokenizerHelper {
            Tagged(Tagged),
            Legacy(serde_json::Value),
        }

        #[derive(Deserialize)]
        #[serde(untagged)]
        pub enum PreTokenizerUntagged {
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

        let helper = PreTokenizerHelper::deserialize(deserializer)?;

        Ok(match helper {
            PreTokenizerHelper::Tagged(pretok) => {
                let mut values: serde_json::Map<String, serde_json::Value> =
                    serde_json::from_value(pretok.rest).map_err(serde::de::Error::custom)?;
                values.insert(
                    "type".to_string(),
                    serde_json::to_value(&pretok.variant).map_err(serde::de::Error::custom)?,
                );
                let values = serde_json::Value::Object(values);
                match pretok.variant {
                    EnumType::BertPreTokenizer => PreTokenizerWrapper::BertPreTokenizer(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::ByteLevel => PreTokenizerWrapper::ByteLevel(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Delimiter => PreTokenizerWrapper::Delimiter(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Metaspace => PreTokenizerWrapper::Metaspace(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Whitespace => PreTokenizerWrapper::Whitespace(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Sequence => PreTokenizerWrapper::Sequence(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Split => PreTokenizerWrapper::Split(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Punctuation => PreTokenizerWrapper::Punctuation(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::WhitespaceSplit => PreTokenizerWrapper::WhitespaceSplit(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Digits => PreTokenizerWrapper::Digits(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::UnicodeScripts => PreTokenizerWrapper::UnicodeScripts(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                }
            }

            PreTokenizerHelper::Legacy(value) => {
                let untagged = serde_json::from_value(value).map_err(serde::de::Error::custom)?;
                match untagged {
                    PreTokenizerUntagged::BertPreTokenizer(bert) => {
                        PreTokenizerWrapper::BertPreTokenizer(bert)
                    }
                    PreTokenizerUntagged::ByteLevel(byte_level) => {
                        PreTokenizerWrapper::ByteLevel(byte_level)
                    }
                    PreTokenizerUntagged::Delimiter(delimiter) => {
                        PreTokenizerWrapper::Delimiter(delimiter)
                    }
                    PreTokenizerUntagged::Metaspace(metaspace) => {
                        PreTokenizerWrapper::Metaspace(metaspace)
                    }
                    PreTokenizerUntagged::Whitespace(whitespace) => {
                        PreTokenizerWrapper::Whitespace(whitespace)
                    }
                    PreTokenizerUntagged::Sequence(sequence) => {
                        PreTokenizerWrapper::Sequence(sequence)
                    }
                    PreTokenizerUntagged::Split(split) => PreTokenizerWrapper::Split(split),
                    PreTokenizerUntagged::Punctuation(punctuation) => {
                        PreTokenizerWrapper::Punctuation(punctuation)
                    }
                    PreTokenizerUntagged::WhitespaceSplit(whitespace_split) => {
                        PreTokenizerWrapper::WhitespaceSplit(whitespace_split)
                    }
                    PreTokenizerUntagged::Digits(digits) => PreTokenizerWrapper::Digits(digits),
                    PreTokenizerUntagged::UnicodeScripts(unicode_scripts) => {
                        PreTokenizerWrapper::UnicodeScripts(unicode_scripts)
                    }
                }
            }
        })
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

    #[test]
    fn pre_tokenizer_deserialization_no_type() {
        let json = r#"{"replacement":"▁","add_prefix_space":true, "prepend_scheme":"always"}}"#;
        let reconstructed = serde_json::from_str::<PreTokenizerWrapper>(json);
        match reconstructed {
            Err(err) => assert_eq!(
                err.to_string(),
                "data did not match any variant of untagged enum PreTokenizerUntagged"
            ),
            _ => panic!("Expected an error here"),
        }

        let json = r#"{"type":"Metaspace", "replacement":"▁" }"#;
        let reconstructed = serde_json::from_str::<PreTokenizerWrapper>(json).unwrap();
        assert_eq!(
            reconstructed,
            PreTokenizerWrapper::Metaspace(Metaspace::default())
        );

        let json = r#"{"type":"Metaspace", "add_prefix_space":true }"#;
        let reconstructed = serde_json::from_str::<PreTokenizerWrapper>(json);
        match reconstructed {
            Err(err) => assert_eq!(err.to_string(), "missing field `replacement`"),
            _ => panic!("Expected an error here"),
        }
        let json = r#"{"behavior":"default_split"}"#;
        let reconstructed = serde_json::from_str::<PreTokenizerWrapper>(json);
        match reconstructed {
            Err(err) => assert_eq!(
                err.to_string(),
                "data did not match any variant of untagged enum PreTokenizerUntagged"
            ),
            _ => panic!("Expected an error here"),
        }
    }
}
