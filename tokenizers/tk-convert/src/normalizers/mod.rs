//! `NormalizerWrapper`: the `normalizer` field of a `tokenizer.json`, as an enum over every
//! normalizer, plus the `Sequence` normalizer that holds a `Vec` of them.
//!
//! As with every wrapper here, the `Deserialize` impl is hand-written so that a tagged config is
//! routed by its tag and an untagged (pre-`"type"`) one still loads.

pub mod mirror;
pub mod utils;

pub use utils::Sequence;

use serde::{Deserialize, Deserializer, Serialize};

use tk_encode::normalizers::bert::BertNormalizer;
use tk_encode::normalizers::byte_level::ByteLevel;
use tk_encode::normalizers::precompiled::Precompiled;
use tk_encode::normalizers::prepend::Prepend;
use tk_encode::normalizers::replace::Replace;
use tk_encode::normalizers::strip::{Strip, StripAccents};
use tk_encode::normalizers::unicode::{NFC, NFD, NFKC, NFKD, Nmt};
use tk_encode::normalizers::utils::Lowercase;
use tk_encode::{NormalizedString, Normalizer, Result, pipeline};

use crate::macros::impl_enum_from;

/// Wrapper for known Normalizers.
///
/// Every variant but two names a mirror in [`mirror`]: `tk-encode` defines the normalizer types and
/// derives serde on none of them, so the on-disk shape of each one is described there. The two
/// exceptions are `Sequence`, which is this crate's own type, and `Precompiled`, whose serde comes
/// from the `spm_precompiled` crate that defines it.
#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum NormalizerWrapper {
    #[serde(with = "mirror::bert")]
    BertNormalizer(BertNormalizer),
    #[serde(with = "mirror::strip")]
    StripNormalizer(Strip),
    #[serde(with = "mirror::strip_accents")]
    StripAccents(StripAccents),
    #[serde(with = "mirror::nfc")]
    NFC(NFC),
    #[serde(with = "mirror::nfd")]
    NFD(NFD),
    #[serde(with = "mirror::nfkc")]
    NFKC(NFKC),
    #[serde(with = "mirror::nfkd")]
    NFKD(NFKD),
    Sequence(Sequence),
    #[serde(with = "mirror::lowercase")]
    Lowercase(Lowercase),
    #[serde(with = "mirror::nmt")]
    Nmt(Nmt),
    Precompiled(Precompiled),
    #[serde(with = "mirror::replace")]
    Replace(Replace),
    #[serde(with = "mirror::PrependDef")]
    Prepend(Prepend),
    #[serde(with = "mirror::byte_level")]
    ByteLevel(ByteLevel),
}

impl<'de> Deserialize<'de> for NormalizerWrapper {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Debug, Deserialize)]
        pub struct Tagged {
            #[serde(rename = "type")]
            variant: EnumType,
            #[serde(flatten)]
            rest: serde_json::Value,
        }
        #[derive(Debug, Serialize, Deserialize)]
        pub enum EnumType {
            Bert,
            Strip,
            StripAccents,
            NFC,
            NFD,
            NFKC,
            NFKD,
            Sequence,
            Lowercase,
            Nmt,
            Precompiled,
            Replace,
            Prepend,
            ByteLevel,
        }

        #[derive(Deserialize)]
        #[serde(untagged)]
        pub enum NormalizerHelper {
            Tagged(Tagged),
            Legacy(serde_json::Value),
        }

        // The legacy fallback, and the reason the `"type"` tag being *required* matters per
        // normalizer: an untagged enum tries each variant in order and takes the first that fits,
        // so a variant lenient about its tag will claim an object that should have been rejected.
        // See `mirror`'s docs for which normalizers require it and why.
        #[derive(Deserialize)]
        #[serde(untagged)]
        pub enum NormalizerUntagged {
            #[serde(with = "mirror::bert")]
            BertNormalizer(BertNormalizer),
            #[serde(with = "mirror::strip")]
            StripNormalizer(Strip),
            #[serde(with = "mirror::strip_accents")]
            StripAccents(StripAccents),
            #[serde(with = "mirror::nfc")]
            NFC(NFC),
            #[serde(with = "mirror::nfd")]
            NFD(NFD),
            #[serde(with = "mirror::nfkc")]
            NFKC(NFKC),
            #[serde(with = "mirror::nfkd")]
            NFKD(NFKD),
            Sequence(Sequence),
            #[serde(with = "mirror::lowercase")]
            Lowercase(Lowercase),
            #[serde(with = "mirror::nmt")]
            Nmt(Nmt),
            Precompiled(Precompiled),
            #[serde(with = "mirror::replace")]
            Replace(Replace),
            #[serde(with = "mirror::PrependDef")]
            Prepend(Prepend),
            #[serde(with = "mirror::byte_level")]
            ByteLevel(ByteLevel),
        }

        let helper = NormalizerHelper::deserialize(deserializer)?;
        Ok(match helper {
            NormalizerHelper::Tagged(model) => {
                let mut values: serde_json::Map<String, serde_json::Value> =
                    serde_json::from_value(model.rest).expect("Parsed values");
                values.insert(
                    "type".to_string(),
                    serde_json::to_value(&model.variant).expect("Reinsert"),
                );
                let values = serde_json::Value::Object(values);
                // Every arm goes through a `mirror` entry point now, because the leaf types no
                // longer implement `Deserialize` themselves -- `serde_json::from_value` would have
                // nothing to call. Two arms are unchanged: `Sequence` is this crate's own type, and
                // `Precompiled` comes with its own serde and its own `from_str` dance.
                match model.variant {
                    EnumType::Bert => NormalizerWrapper::BertNormalizer(
                        mirror::bert::deserialize(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Strip => NormalizerWrapper::StripNormalizer(
                        mirror::strip::deserialize(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::StripAccents => NormalizerWrapper::StripAccents(
                        mirror::strip_accents::deserialize(values)
                            .map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::NFC => NormalizerWrapper::NFC(
                        mirror::nfc::deserialize(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::NFD => NormalizerWrapper::NFD(
                        mirror::nfd::deserialize(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::NFKC => NormalizerWrapper::NFKC(
                        mirror::nfkc::deserialize(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::NFKD => NormalizerWrapper::NFKD(
                        mirror::nfkd::deserialize(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Sequence => NormalizerWrapper::Sequence(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Lowercase => NormalizerWrapper::Lowercase(
                        mirror::lowercase::deserialize(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Nmt => NormalizerWrapper::Nmt(
                        mirror::nmt::deserialize(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Precompiled => NormalizerWrapper::Precompiled(
                        serde_json::from_str(
                            &serde_json::to_string(&values).expect("Can reserialize precompiled"),
                        )
                        // .map_err(serde::de::Error::custom)
                        .expect("Precompiled"),
                    ),
                    EnumType::Replace => NormalizerWrapper::Replace(
                        mirror::replace::deserialize(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Prepend => NormalizerWrapper::Prepend(
                        mirror::PrependDef::deserialize(values)
                            .map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::ByteLevel => NormalizerWrapper::ByteLevel(
                        mirror::byte_level::deserialize(values)
                            .map_err(serde::de::Error::custom)?,
                    ),
                }
            }

            NormalizerHelper::Legacy(value) => {
                let untagged = serde_json::from_value(value).map_err(serde::de::Error::custom)?;
                match untagged {
                    NormalizerUntagged::BertNormalizer(bpe) => {
                        NormalizerWrapper::BertNormalizer(bpe)
                    }
                    NormalizerUntagged::StripNormalizer(bpe) => {
                        NormalizerWrapper::StripNormalizer(bpe)
                    }
                    NormalizerUntagged::StripAccents(bpe) => NormalizerWrapper::StripAccents(bpe),
                    NormalizerUntagged::NFC(bpe) => NormalizerWrapper::NFC(bpe),
                    NormalizerUntagged::NFD(bpe) => NormalizerWrapper::NFD(bpe),
                    NormalizerUntagged::NFKC(bpe) => NormalizerWrapper::NFKC(bpe),
                    NormalizerUntagged::NFKD(bpe) => NormalizerWrapper::NFKD(bpe),
                    NormalizerUntagged::Sequence(seq) => NormalizerWrapper::Sequence(seq),
                    NormalizerUntagged::Lowercase(bpe) => NormalizerWrapper::Lowercase(bpe),
                    NormalizerUntagged::Nmt(bpe) => NormalizerWrapper::Nmt(bpe),
                    NormalizerUntagged::Precompiled(bpe) => NormalizerWrapper::Precompiled(bpe),
                    NormalizerUntagged::Replace(bpe) => NormalizerWrapper::Replace(bpe),
                    NormalizerUntagged::Prepend(bpe) => NormalizerWrapper::Prepend(bpe),
                    NormalizerUntagged::ByteLevel(bpe) => NormalizerWrapper::ByteLevel(bpe),
                }
            }
        })
    }
}

impl Normalizer for NormalizerWrapper {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        match self {
            Self::BertNormalizer(bn) => bn.normalize(normalized),
            Self::StripNormalizer(sn) => sn.normalize(normalized),
            Self::StripAccents(sn) => sn.normalize(normalized),
            Self::NFC(nfc) => nfc.normalize(normalized),
            Self::NFD(nfd) => nfd.normalize(normalized),
            Self::NFKC(nfkc) => nfkc.normalize(normalized),
            Self::NFKD(nfkd) => nfkd.normalize(normalized),
            Self::Sequence(sequence) => sequence.normalize(normalized),
            Self::Lowercase(lc) => lc.normalize(normalized),
            Self::Nmt(lc) => lc.normalize(normalized),
            Self::Precompiled(lc) => lc.normalize(normalized),
            Self::Replace(lc) => lc.normalize(normalized),
            Self::Prepend(lc) => lc.normalize(normalized),
            Self::ByteLevel(lc) => lc.normalize(normalized),
        }
    }
}

impl_enum_from!(BertNormalizer, NormalizerWrapper, BertNormalizer);
impl_enum_from!(NFKD, NormalizerWrapper, NFKD);
impl_enum_from!(NFKC, NormalizerWrapper, NFKC);
impl_enum_from!(NFC, NormalizerWrapper, NFC);
impl_enum_from!(NFD, NormalizerWrapper, NFD);
impl_enum_from!(Strip, NormalizerWrapper, StripNormalizer);
impl_enum_from!(StripAccents, NormalizerWrapper, StripAccents);
impl_enum_from!(Sequence, NormalizerWrapper, Sequence);
impl_enum_from!(Lowercase, NormalizerWrapper, Lowercase);
impl_enum_from!(Nmt, NormalizerWrapper, Nmt);
impl_enum_from!(Precompiled, NormalizerWrapper, Precompiled);
impl_enum_from!(Replace, NormalizerWrapper, Replace);
impl_enum_from!(Prepend, NormalizerWrapper, Prepend);
impl_enum_from!(ByteLevel, NormalizerWrapper, ByteLevel);

impl pipeline::Normalizer for NormalizerWrapper {
    fn normalize<'a>(&self, input: &'a str) -> Result<std::borrow::Cow<'a, str>> {
        match self {
            Self::BertNormalizer(bn) => pipeline::Normalizer::normalize(bn, input),
            Self::StripNormalizer(sn) => pipeline::Normalizer::normalize(sn, input),
            Self::StripAccents(sn) => pipeline::Normalizer::normalize(sn, input),
            Self::NFC(nfc) => pipeline::Normalizer::normalize(nfc, input),
            Self::NFD(nfd) => pipeline::Normalizer::normalize(nfd, input),
            Self::NFKC(nfkc) => pipeline::Normalizer::normalize(nfkc, input),
            Self::NFKD(nfkd) => pipeline::Normalizer::normalize(nfkd, input),
            Self::Sequence(sequence) => pipeline::Normalizer::normalize(sequence, input),
            Self::Lowercase(lc) => pipeline::Normalizer::normalize(lc, input),
            Self::Nmt(nmt) => pipeline::Normalizer::normalize(nmt, input),
            Self::Precompiled(pc) => pipeline::Normalizer::normalize(pc, input),
            Self::Replace(rp) => pipeline::Normalizer::normalize(rp, input),
            Self::Prepend(pp) => pipeline::Normalizer::normalize(pp, input),
            Self::ByteLevel(bl) => pipeline::Normalizer::normalize(bl, input),
        }
    }
}

// Every test in here round-trips a wrapper through serde, so the module goes with `config`.
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn post_processor_deserialization_no_type() {
        let json = r#"{"strip_left":false, "strip_right":true}"#;
        let reconstructed = serde_json::from_str::<NormalizerWrapper>(json);
        assert!(matches!(
            reconstructed.unwrap(),
            NormalizerWrapper::StripNormalizer(_)
        ));

        let json = r#"{"trim_offsets":true, "add_prefix_space":true}"#;
        let reconstructed = serde_json::from_str::<NormalizerWrapper>(json);
        match reconstructed {
            Err(err) => assert_eq!(
                err.to_string(),
                "data did not match any variant of untagged enum NormalizerUntagged"
            ),
            _ => panic!("Expected an error here"),
        }

        let json = r#"{"prepend":"a"}"#;
        let reconstructed = serde_json::from_str::<NormalizerWrapper>(json);
        assert!(matches!(
            reconstructed.unwrap(),
            NormalizerWrapper::Prepend(_)
        ));
    }

    #[test]
    fn normalizer_serialization() {
        let json = r#"{"type":"Sequence","normalizers":[]}"#;
        assert!(serde_json::from_str::<NormalizerWrapper>(json).is_ok());
        let json = r#"{"type":"Sequence","normalizers":[{}]}"#;
        let parse = serde_json::from_str::<NormalizerWrapper>(json);
        match parse {
            Err(err) => assert_eq!(
                format!("{err}"),
                "data did not match any variant of untagged enum NormalizerUntagged"
            ),
            _ => panic!("Expected error"),
        }

        let json = r#"{"replacement":"▁","prepend_scheme":"always"}"#;
        let parse = serde_json::from_str::<NormalizerWrapper>(json);
        match parse {
            Err(err) => assert_eq!(
                format!("{err}"),
                "data did not match any variant of untagged enum NormalizerUntagged"
            ),
            _ => panic!("Expected error"),
        }

        let json = r#"{"type":"Sequence","prepend_scheme":"always"}"#;
        let parse = serde_json::from_str::<NormalizerWrapper>(json);
        match parse {
            Err(err) => assert_eq!(format!("{err}"), "missing field `normalizers`"),
            _ => panic!("Expected error"),
        }
    }
}
