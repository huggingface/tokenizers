pub mod bert;
pub mod byte_level;
#[cfg(feature = "spm")]
pub mod precompiled;
pub mod prepend;
pub mod replace;
pub mod strip;
pub mod unicode;
pub mod utils;
pub use crate::normalizers::bert::BertNormalizer;
pub use crate::normalizers::byte_level::ByteLevel;
#[cfg(feature = "spm")]
pub use crate::normalizers::precompiled::Precompiled;
pub use crate::normalizers::prepend::Prepend;
pub use crate::normalizers::replace::Replace;
pub use crate::normalizers::strip::{Strip, StripAccents};
pub use crate::normalizers::unicode::Nmt;
#[cfg(feature = "unicode-normalization")]
pub use crate::normalizers::unicode::{NFC, NFD, NFKC, NFKD};
pub use crate::normalizers::utils::{Lowercase, Sequence};
use serde::{Deserialize, Deserializer, Serialize};

use crate::{NormalizedString, Normalizer};

/// Wrapper for known Normalizers.
#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum NormalizerWrapper {
    BertNormalizer(BertNormalizer),
    StripNormalizer(Strip),
    StripAccents(StripAccents),
    #[cfg(feature = "unicode-normalization")]
    NFC(NFC),
    #[cfg(feature = "unicode-normalization")]
    NFD(NFD),
    #[cfg(feature = "unicode-normalization")]
    NFKC(NFKC),
    #[cfg(feature = "unicode-normalization")]
    NFKD(NFKD),
    Sequence(Sequence),
    Lowercase(Lowercase),
    Nmt(Nmt),
    #[cfg(feature = "spm")]
    Precompiled(Precompiled),
    Replace(Replace),
    Prepend(Prepend),
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

        #[derive(Deserialize)]
        #[serde(untagged)]
        pub enum NormalizerUntagged {
            BertNormalizer(BertNormalizer),
            StripNormalizer(Strip),
            StripAccents(StripAccents),
            #[cfg(feature = "unicode-normalization")]
            NFC(NFC),
            #[cfg(feature = "unicode-normalization")]
            NFD(NFD),
            #[cfg(feature = "unicode-normalization")]
            NFKC(NFKC),
            #[cfg(feature = "unicode-normalization")]
            NFKD(NFKD),
            Sequence(Sequence),
            Lowercase(Lowercase),
            Nmt(Nmt),
            Replace(Replace),
            Prepend(Prepend),
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
                match model.variant {
                    EnumType::Bert => NormalizerWrapper::BertNormalizer(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Strip => NormalizerWrapper::StripNormalizer(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::StripAccents => NormalizerWrapper::StripAccents(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::NFC => {
                        #[cfg(feature = "unicode-normalization")]
                        {
                            NormalizerWrapper::NFC(
                                serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                            )
                        }
                        #[cfg(not(feature = "unicode-normalization"))]
                        {
                            return Err(serde::de::Error::custom(
                                "NFC normalizer requires the `unicode-normalization` feature",
                            ));
                        }
                    }
                    EnumType::NFD => {
                        #[cfg(feature = "unicode-normalization")]
                        {
                            NormalizerWrapper::NFD(
                                serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                            )
                        }
                        #[cfg(not(feature = "unicode-normalization"))]
                        {
                            return Err(serde::de::Error::custom(
                                "NFD normalizer requires the `unicode-normalization` feature",
                            ));
                        }
                    }
                    EnumType::NFKC => {
                        #[cfg(feature = "unicode-normalization")]
                        {
                            NormalizerWrapper::NFKC(
                                serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                            )
                        }
                        #[cfg(not(feature = "unicode-normalization"))]
                        {
                            return Err(serde::de::Error::custom(
                                "NFKC normalizer requires the `unicode-normalization` feature",
                            ));
                        }
                    }
                    EnumType::NFKD => {
                        #[cfg(feature = "unicode-normalization")]
                        {
                            NormalizerWrapper::NFKD(
                                serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                            )
                        }
                        #[cfg(not(feature = "unicode-normalization"))]
                        {
                            return Err(serde::de::Error::custom(
                                "NFKD normalizer requires the `unicode-normalization` feature",
                            ));
                        }
                    }
                    EnumType::Sequence => NormalizerWrapper::Sequence(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Lowercase => NormalizerWrapper::Lowercase(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Nmt => NormalizerWrapper::Nmt(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Precompiled => {
                        #[cfg(feature = "spm")]
                        {
                            NormalizerWrapper::Precompiled(
                                serde_json::from_str(
                                    &serde_json::to_string(&values)
                                        .expect("Can reserialize precompiled"),
                                )
                                .expect("Precompiled"),
                            )
                        }
                        #[cfg(not(feature = "spm"))]
                        {
                            return Err(serde::de::Error::custom(
                                "Precompiled normalizer requires the `spm` feature",
                            ));
                        }
                    }
                    EnumType::Replace => NormalizerWrapper::Replace(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Prepend => NormalizerWrapper::Prepend(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::ByteLevel => NormalizerWrapper::ByteLevel(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
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
                    #[cfg(feature = "unicode-normalization")]
                    NormalizerUntagged::NFC(bpe) => NormalizerWrapper::NFC(bpe),
                    #[cfg(feature = "unicode-normalization")]
                    NormalizerUntagged::NFD(bpe) => NormalizerWrapper::NFD(bpe),
                    #[cfg(feature = "unicode-normalization")]
                    NormalizerUntagged::NFKC(bpe) => NormalizerWrapper::NFKC(bpe),
                    #[cfg(feature = "unicode-normalization")]
                    NormalizerUntagged::NFKD(bpe) => NormalizerWrapper::NFKD(bpe),
                    NormalizerUntagged::Sequence(seq) => NormalizerWrapper::Sequence(seq),
                    NormalizerUntagged::Lowercase(bpe) => NormalizerWrapper::Lowercase(bpe),
                    NormalizerUntagged::Nmt(bpe) => NormalizerWrapper::Nmt(bpe),
                    NormalizerUntagged::Replace(bpe) => NormalizerWrapper::Replace(bpe),
                    NormalizerUntagged::Prepend(bpe) => NormalizerWrapper::Prepend(bpe),
                    NormalizerUntagged::ByteLevel(bpe) => NormalizerWrapper::ByteLevel(bpe),
                }
            }
        })
    }
}

impl Normalizer for NormalizerWrapper {
    fn normalize(&self, normalized: &mut NormalizedString) -> crate::Result<()> {
        match self {
            Self::BertNormalizer(bn) => bn.normalize(normalized),
            Self::StripNormalizer(sn) => sn.normalize(normalized),
            Self::StripAccents(sn) => sn.normalize(normalized),
            #[cfg(feature = "unicode-normalization")]
            Self::NFC(nfc) => nfc.normalize(normalized),
            #[cfg(feature = "unicode-normalization")]
            Self::NFD(nfd) => nfd.normalize(normalized),
            #[cfg(feature = "unicode-normalization")]
            Self::NFKC(nfkc) => nfkc.normalize(normalized),
            #[cfg(feature = "unicode-normalization")]
            Self::NFKD(nfkd) => nfkd.normalize(normalized),
            Self::Sequence(sequence) => sequence.normalize(normalized),
            Self::Lowercase(lc) => lc.normalize(normalized),
            Self::Nmt(lc) => lc.normalize(normalized),
            #[cfg(feature = "spm")]
            Self::Precompiled(lc) => lc.normalize(normalized),
            Self::Replace(lc) => lc.normalize(normalized),
            Self::Prepend(lc) => lc.normalize(normalized),
            Self::ByteLevel(lc) => lc.normalize(normalized),
        }
    }
}

impl_enum_from!(BertNormalizer, NormalizerWrapper, BertNormalizer);
#[cfg(feature = "unicode-normalization")]
impl_enum_from!(NFKD, NormalizerWrapper, NFKD);
#[cfg(feature = "unicode-normalization")]
impl_enum_from!(NFKC, NormalizerWrapper, NFKC);
#[cfg(feature = "unicode-normalization")]
impl_enum_from!(NFC, NormalizerWrapper, NFC);
#[cfg(feature = "unicode-normalization")]
impl_enum_from!(NFD, NormalizerWrapper, NFD);
impl_enum_from!(Strip, NormalizerWrapper, StripNormalizer);
impl_enum_from!(StripAccents, NormalizerWrapper, StripAccents);
impl_enum_from!(Sequence, NormalizerWrapper, Sequence);
impl_enum_from!(Lowercase, NormalizerWrapper, Lowercase);
impl_enum_from!(Nmt, NormalizerWrapper, Nmt);
#[cfg(feature = "spm")]
impl_enum_from!(Precompiled, NormalizerWrapper, Precompiled);
impl_enum_from!(Replace, NormalizerWrapper, Replace);
impl_enum_from!(Prepend, NormalizerWrapper, Prepend);
impl_enum_from!(ByteLevel, NormalizerWrapper, ByteLevel);

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
