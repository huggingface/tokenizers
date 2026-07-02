pub mod bpe;
pub mod byte_fallback;
pub mod ctc;
pub mod fuse;
pub mod sequence;
pub mod strip;
pub mod wordpiece;

// Re-export these as decoders
pub use super::pre_tokenizers::byte_level;
pub use super::pre_tokenizers::metaspace;

use serde::{Deserialize, Deserializer, Serialize};

use crate::decoders::bpe::BPEDecoder;
use crate::decoders::byte_fallback::ByteFallback;
use crate::decoders::ctc::CTC;
use crate::decoders::fuse::Fuse;
use crate::decoders::sequence::Sequence;
use crate::decoders::strip::Strip;
use crate::decoders::wordpiece::WordPiece;
use crate::normalizers::replace::Replace;
use crate::pre_tokenizers::byte_level::ByteLevel;
use crate::pre_tokenizers::metaspace::Metaspace;
use crate::{Decoder, Result};

#[derive(Serialize, Clone, Debug)]
#[serde(untagged)]
pub enum DecoderWrapper {
    BPE(BPEDecoder),
    ByteLevel(ByteLevel),
    WordPiece(WordPiece),
    Metaspace(Metaspace),
    CTC(CTC),
    Sequence(Sequence),
    Replace(Replace),
    Fuse(Fuse),
    Strip(Strip),
    ByteFallback(ByteFallback),
}

impl<'de> Deserialize<'de> for DecoderWrapper {
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
        #[derive(Serialize, Deserialize)]
        pub enum EnumType {
            BPEDecoder,
            ByteLevel,
            WordPiece,
            Metaspace,
            CTC,
            Sequence,
            Replace,
            Fuse,
            Strip,
            ByteFallback,
        }

        #[derive(Deserialize)]
        #[serde(untagged)]
        pub enum DecoderHelper {
            Tagged(Tagged),
            Legacy(serde_json::Value),
        }

        #[derive(Deserialize)]
        #[serde(untagged)]
        pub enum DecoderUntagged {
            BPE(BPEDecoder),
            ByteLevel(ByteLevel),
            WordPiece(WordPiece),
            Metaspace(Metaspace),
            CTC(CTC),
            Sequence(Sequence),
            Replace(Replace),
            Fuse(Fuse),
            Strip(Strip),
            ByteFallback(ByteFallback),
        }

        let helper = DecoderHelper::deserialize(deserializer).expect("Helper");
        Ok(match helper {
            DecoderHelper::Tagged(model) => {
                let mut values: serde_json::Map<String, serde_json::Value> =
                    serde_json::from_value(model.rest).map_err(serde::de::Error::custom)?;
                values.insert(
                    "type".to_string(),
                    serde_json::to_value(&model.variant).map_err(serde::de::Error::custom)?,
                );
                let values = serde_json::Value::Object(values);
                match model.variant {
                    EnumType::BPEDecoder => DecoderWrapper::BPE(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::ByteLevel => DecoderWrapper::ByteLevel(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::WordPiece => DecoderWrapper::WordPiece(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Metaspace => DecoderWrapper::Metaspace(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::CTC => DecoderWrapper::CTC(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Sequence => DecoderWrapper::Sequence(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Replace => DecoderWrapper::Replace(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Fuse => DecoderWrapper::Fuse(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Strip => DecoderWrapper::Strip(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::ByteFallback => DecoderWrapper::ByteFallback(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                }
            }
            DecoderHelper::Legacy(value) => {
                let untagged = serde_json::from_value(value).map_err(serde::de::Error::custom)?;
                match untagged {
                    DecoderUntagged::BPE(dec) => DecoderWrapper::BPE(dec),
                    DecoderUntagged::ByteLevel(dec) => DecoderWrapper::ByteLevel(dec),
                    DecoderUntagged::WordPiece(dec) => DecoderWrapper::WordPiece(dec),
                    DecoderUntagged::Metaspace(dec) => DecoderWrapper::Metaspace(dec),
                    DecoderUntagged::CTC(dec) => DecoderWrapper::CTC(dec),
                    DecoderUntagged::Sequence(dec) => DecoderWrapper::Sequence(dec),
                    DecoderUntagged::Replace(dec) => DecoderWrapper::Replace(dec),
                    DecoderUntagged::Fuse(dec) => DecoderWrapper::Fuse(dec),
                    DecoderUntagged::Strip(dec) => DecoderWrapper::Strip(dec),
                    DecoderUntagged::ByteFallback(dec) => DecoderWrapper::ByteFallback(dec),
                }
            }
        })
    }
}

impl Decoder for DecoderWrapper {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        match self {
            Self::BPE(bpe) => bpe.decode_chain(tokens),
            Self::ByteLevel(bl) => bl.decode_chain(tokens),
            Self::Metaspace(ms) => ms.decode_chain(tokens),
            Self::WordPiece(wp) => wp.decode_chain(tokens),
            Self::CTC(ctc) => ctc.decode_chain(tokens),
            Self::Sequence(seq) => seq.decode_chain(tokens),
            Self::Replace(seq) => seq.decode_chain(tokens),
            Self::ByteFallback(bf) => bf.decode_chain(tokens),
            Self::Strip(bf) => bf.decode_chain(tokens),
            Self::Fuse(bf) => bf.decode_chain(tokens),
        }
    }
}

impl_enum_from!(BPEDecoder, DecoderWrapper, BPE);
impl_enum_from!(ByteLevel, DecoderWrapper, ByteLevel);
impl_enum_from!(ByteFallback, DecoderWrapper, ByteFallback);
impl_enum_from!(Fuse, DecoderWrapper, Fuse);
impl_enum_from!(Strip, DecoderWrapper, Strip);
impl_enum_from!(Metaspace, DecoderWrapper, Metaspace);
impl_enum_from!(WordPiece, DecoderWrapper, WordPiece);
impl_enum_from!(CTC, DecoderWrapper, CTC);
impl_enum_from!(Sequence, DecoderWrapper, Sequence);
impl_enum_from!(Replace, DecoderWrapper, Replace);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_serialization() {
        let oldjson = r#"{"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Metaspace","replacement":"▁","add_prefix_space":true,"prepend_scheme":"always"}]}"#;
        let olddecoder: DecoderWrapper = serde_json::from_str(oldjson).unwrap();
        let oldserialized = serde_json::to_string(&olddecoder).unwrap();
        let json = r#"{"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}]}"#;
        assert_eq!(oldserialized, json);

        let decoder: DecoderWrapper = serde_json::from_str(json).unwrap();
        let serialized = serde_json::to_string(&decoder).unwrap();
        assert_eq!(serialized, json);
    }
    #[test]
    fn decoder_serialization_other_no_arg() {
        let json = r#"{"type":"Sequence","decoders":[{"type":"Fuse"},{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}]}"#;
        let decoder: DecoderWrapper = serde_json::from_str(json).unwrap();
        let serialized = serde_json::to_string(&decoder).unwrap();
        assert_eq!(serialized, json);
    }

    #[test]
    fn decoder_serialization_no_decode() {
        let json = r#"{"type":"Sequence","decoders":[{},{"type":"Metaspace","replacement":"▁","prepend_scheme":"always"}]}"#;
        let parse = serde_json::from_str::<DecoderWrapper>(json);
        match parse {
            Err(err) => assert_eq!(
                format!("{err}"),
                "data did not match any variant of untagged enum DecoderUntagged"
            ),
            _ => panic!("Expected error"),
        }

        let json = r#"{"replacement":"▁","prepend_scheme":"always"}"#;
        let parse = serde_json::from_str::<DecoderWrapper>(json);
        match parse {
            Err(err) => assert_eq!(
                format!("{err}"),
                "data did not match any variant of untagged enum DecoderUntagged"
            ),
            _ => panic!("Expected error"),
        }

        let json = r#"{"type":"Sequence","prepend_scheme":"always"}"#;
        let parse = serde_json::from_str::<DecoderWrapper>(json);
        match parse {
            Err(err) => assert_eq!(format!("{err}"), "missing field `decoders`"),
            _ => panic!("Expected error"),
        }
    }
}
