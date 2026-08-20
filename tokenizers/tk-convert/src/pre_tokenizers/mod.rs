//! `PreTokenizerWrapper`: the `pre_tokenizer` field of a `tokenizer.json`, as an enum over every
//! pre-tokenizer, plus the `Sequence` pre-tokenizer that holds a `Vec` of them.

pub mod mirror;
pub mod sequence;

pub use sequence::Sequence;

use serde::{Deserialize, Deserializer, Serialize};

use tk_encode::pre_tokenizers::bert::BertPreTokenizer;
use tk_encode::pre_tokenizers::byte_level::ByteLevel;
use tk_encode::pre_tokenizers::delimiter::CharDelimiterSplit;
use tk_encode::pre_tokenizers::digits::Digits;
use tk_encode::pre_tokenizers::fixed_length::FixedLength;
use tk_encode::pre_tokenizers::metaspace::Metaspace;
use tk_encode::pre_tokenizers::punctuation::Punctuation;
use tk_encode::pre_tokenizers::split::Split;
use tk_encode::pre_tokenizers::unicode_scripts::UnicodeScripts;
use tk_encode::pre_tokenizers::whitespace::{Whitespace, WhitespaceSplit};
use tk_encode::{PreTokenizedString, PreTokenizer, Result};

use crate::macros::impl_enum_from;

/// Every variant now names a mirror in [`mirror`]: `tk-encode` defines the twelve pre-tokenizer
/// types and derives serde on none of them, so the on-disk shape of each one is described there.
/// `Sequence` is the exception, being this crate's own type.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
#[allow(clippy::large_enum_variant)] // Split holds a compiled regex; boxing it would churn the API
pub enum PreTokenizerWrapper {
    #[serde(with = "mirror::bert")]
    BertPreTokenizer(BertPreTokenizer),
    #[serde(with = "mirror::byte_level")]
    ByteLevel(ByteLevel),
    #[serde(with = "mirror::delimiter")]
    Delimiter(CharDelimiterSplit),
    #[serde(with = "mirror::metaspace")]
    Metaspace(Metaspace),
    #[serde(with = "mirror::whitespace")]
    Whitespace(Whitespace),
    Sequence(Sequence),
    #[serde(with = "mirror::split")]
    Split(Split),
    #[serde(with = "mirror::punctuation")]
    Punctuation(Punctuation),
    #[serde(with = "mirror::whitespace_split")]
    WhitespaceSplit(WhitespaceSplit),
    #[serde(with = "mirror::digits")]
    Digits(Digits),
    #[serde(with = "mirror::unicode_scripts")]
    UnicodeScripts(UnicodeScripts),
    #[serde(with = "mirror::fixed_length")]
    FixedLength(FixedLength),
}

impl PreTokenizer for PreTokenizerWrapper {
    fn pre_tokenize(&self, normalized: &mut PreTokenizedString) -> Result<()> {
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
            Self::FixedLength(fl) => fl.pre_tokenize(normalized),
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
            FixedLength,
        }

        #[derive(Deserialize)]
        #[serde(untagged)]
        pub enum PreTokenizerHelper {
            Tagged(Tagged),
            Legacy(serde_json::Value),
        }

        #[derive(Deserialize)]
        #[serde(untagged)]
        #[allow(clippy::large_enum_variant)]
        pub enum PreTokenizerUntagged {
            #[serde(with = "mirror::bert")]
            BertPreTokenizer(BertPreTokenizer),
            #[serde(with = "mirror::byte_level")]
            ByteLevel(ByteLevel),
            #[serde(with = "mirror::delimiter")]
            Delimiter(CharDelimiterSplit),
            #[serde(with = "mirror::metaspace")]
            Metaspace(Metaspace),
            #[serde(with = "mirror::whitespace")]
            Whitespace(Whitespace),
            Sequence(Sequence),
            #[serde(with = "mirror::split")]
            Split(Split),
            #[serde(with = "mirror::punctuation")]
            Punctuation(Punctuation),
            #[serde(with = "mirror::whitespace_split")]
            WhitespaceSplit(WhitespaceSplit),
            #[serde(with = "mirror::digits")]
            Digits(Digits),
            #[serde(with = "mirror::unicode_scripts")]
            UnicodeScripts(UnicodeScripts),
            #[serde(with = "mirror::fixed_length")]
            FixedLength(FixedLength),
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
                // Every arm goes through a `mirror` entry point now, because the leaf types no
                // longer implement `Deserialize` themselves -- `serde_json::from_value` would have
                // nothing to call. `Sequence` is the exception: it is this crate's own type.
                //
                // Note `EnumType::Delimiter` re-inserts `"type":"Delimiter"`, which
                // `mirror::delimiter` rejects: the tag `CharDelimiterSplit` writes and accepts is
                // its own name. That was already true of `from_value::<CharDelimiterSplit>` and is
                // deliberately left alone -- such a config loads through the untagged fallback
                // below, and "fixing" it here would start accepting a spelling nothing emits.
                match pretok.variant {
                    EnumType::BertPreTokenizer => PreTokenizerWrapper::BertPreTokenizer(
                        mirror::bert::deserialize(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::ByteLevel => PreTokenizerWrapper::ByteLevel(
                        mirror::byte_level::deserialize(values)
                            .map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Delimiter => PreTokenizerWrapper::Delimiter(
                        mirror::delimiter::deserialize(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Metaspace => PreTokenizerWrapper::Metaspace(
                        mirror::metaspace::deserialize(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Whitespace => PreTokenizerWrapper::Whitespace(
                        mirror::whitespace::deserialize(values)
                            .map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Sequence => PreTokenizerWrapper::Sequence(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Split => PreTokenizerWrapper::Split(
                        mirror::split::deserialize(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Punctuation => PreTokenizerWrapper::Punctuation(
                        mirror::punctuation::deserialize(values)
                            .map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::WhitespaceSplit => PreTokenizerWrapper::WhitespaceSplit(
                        mirror::whitespace_split::deserialize(values)
                            .map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Digits => PreTokenizerWrapper::Digits(
                        mirror::digits::deserialize(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::UnicodeScripts => PreTokenizerWrapper::UnicodeScripts(
                        mirror::unicode_scripts::deserialize(values)
                            .map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::FixedLength => PreTokenizerWrapper::FixedLength(
                        mirror::fixed_length::deserialize(values)
                            .map_err(serde::de::Error::custom)?,
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
                    PreTokenizerUntagged::FixedLength(fixed_length) => {
                        PreTokenizerWrapper::FixedLength(fixed_length)
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
impl_enum_from!(FixedLength, PreTokenizerWrapper, FixedLength);

// Every test in here round-trips a wrapper through serde, so the module goes with `config`.
#[cfg(test)]
mod tests {
    use super::*;
    use tk_encode::pre_tokenizers::metaspace::PrependScheme;

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
            PreTokenizerWrapper::Metaspace(Metaspace::new('▁', PrependScheme::First, true))
        );

        let pre_tokenizer: PreTokenizerWrapper = serde_json::from_str(
            r#"{"type":"Metaspace","replacement":"▁","add_prefix_space":true, "prepend_scheme":"always"}"#,
        )
        .unwrap();

        assert_eq!(
            pre_tokenizer,
            PreTokenizerWrapper::Metaspace(Metaspace::new('▁', PrependScheme::Always, true))
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
