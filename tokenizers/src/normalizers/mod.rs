pub mod bert;
pub mod precompiled;
pub mod prepend;
pub mod replace;
pub mod strip;
pub mod unicode;
pub mod utils;

pub use crate::normalizers::bert::BertNormalizer;
pub use crate::normalizers::precompiled::Precompiled;
pub use crate::normalizers::prepend::Prepend;
pub use crate::normalizers::replace::Replace;
pub use crate::normalizers::strip::{Strip, StripAccents};
pub use crate::normalizers::unicode::{Nmt, NFC, NFD, NFKC, NFKD};
pub use crate::normalizers::utils::{Lowercase, Sequence};

use serde::{Deserialize, Serialize};

use crate::{NormalizedString, Normalizer};

/// Wrapper for known Normalizers.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum NormalizerWrapper {
    BertNormalizer(BertNormalizer),
    StripNormalizer(Strip),
    StripAccents(StripAccents),
    NFC(NFC),
    NFD(NFD),
    NFKC(NFKC),
    NFKD(NFKD),
    Sequence(Sequence),
    Lowercase(Lowercase),
    Nmt(Nmt),
    Precompiled(Precompiled),
    Replace(Replace),
    Prepend(Prepend),
}

#[cfg(feature = "arbitrary")]
impl<'a> arbitrary::Arbitrary<'a> for NormalizerWrapper {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(match u8::arbitrary(u)? {
            0 => NormalizerWrapper::BertNormalizer(BertNormalizer::arbitrary(u)?),
            1 => NormalizerWrapper::StripNormalizer(Strip::arbitrary(u)?),
            2 => NormalizerWrapper::StripAccents(StripAccents::arbitrary(u)?),
            3 => NormalizerWrapper::NFC(NFC::arbitrary(u)?),
            4 => NormalizerWrapper::NFD(NFD::arbitrary(u)?),
            5 => NormalizerWrapper::NFKC(NFKC::arbitrary(u)?),
            6 => NormalizerWrapper::NFKD(NFKD::arbitrary(u)?),
            7 => NormalizerWrapper::Sequence(Sequence::arbitrary(u)?),
            8 => NormalizerWrapper::Lowercase(Lowercase::arbitrary(u)?),
            9 => NormalizerWrapper::Nmt(Nmt::arbitrary(u)?),
            10 => NormalizerWrapper::Precompiled(
                Precompiled::from(<&[u8]>::arbitrary(u)?)
                    .map_err(|_| arbitrary::Error::IncorrectFormat)?,
            ),
            11 => NormalizerWrapper::Replace(Replace::arbitrary(u)?),
            _ => NormalizerWrapper::Prepend(Prepend::arbitrary(u)?),
        })
    }
}

impl Normalizer for NormalizerWrapper {
    fn normalize(&self, normalized: &mut NormalizedString) -> crate::Result<()> {
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
