#[cfg(feature = "normalizers")]
pub mod bert;
pub mod byte_level;
pub mod metaspace;
#[cfg(feature = "normalizers")]
pub mod precompiled;
pub mod prepend;
pub mod replace;
#[cfg(feature = "serde")]
mod serialization;
pub mod strip;
#[cfg(feature = "normalizers")]
pub mod unicode;
pub mod utils;
#[cfg(feature = "normalizers")]
pub use crate::normalizers::bert::BertNormalizer;
pub use crate::normalizers::byte_level::ByteLevel;
#[cfg(feature = "normalizers")]
pub use crate::normalizers::precompiled::Precompiled;
pub use crate::normalizers::prepend::Prepend;
pub use crate::normalizers::replace::Replace;
pub use crate::normalizers::strip::Strip;
#[cfg(feature = "normalizers")]
pub use crate::normalizers::strip::StripAccents;
#[cfg(feature = "normalizers")]
pub use crate::normalizers::unicode::{NFC, NFD, NFKC, NFKD, Nmt};
pub use crate::normalizers::utils::Lowercase;

// `NormalizerWrapper`, its hand-written `Deserialize` (tagged, with an untagged legacy fallback)
// and the `Sequence` normalizer that held a `Vec<NormalizerWrapper>` are gone, deleted with the
// config layer. What the encode path runs is `pipeline::PipelineNormalizer`, one variant per
// concrete normalizer: the reader flattens a config `Sequence` into a `Vec` of those, applied
// through `pipeline::NormalizerChain`.
