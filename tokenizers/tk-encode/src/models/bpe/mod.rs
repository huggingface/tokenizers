//! [Byte Pair Encoding](https://www.aclweb.org/anthology/P16-1162/) model.
use ahash::AHashMap;

mod convert;
mod fold;
mod merge_hot_cold_queue;
mod merge_multipass;
mod model;
mod pair_map;
mod serialization;
mod tables;

#[cfg(test)]
mod tests;

pub type Pair = (u32, u32);
/// Token string -> external id, as the model file declares it.
pub type Vocab = AHashMap<String, u32>;
/// External id -> token string.
pub type VocabR = AHashMap<u32, String>;
/// Merge pairs as token strings, highest priority first.
pub type Merges = Vec<(String, String)>;
/// Merge pair (external ids) -> (rank, external id of the merged token).
pub type MergeMap = AHashMap<Pair, (u32, u32)>;

/// The word cache's default size, and the longest word it will hold.
///
/// Re-exported from `utils::cache`, which is `pub(crate)`: the config-shaped `BPE` in `tk-convert`
/// keeps a cache of its own and needs both numbers, and a second copy of them on that side would be
/// a second answer to "how big is a BPE cache".
pub use crate::utils::cache::{DEFAULT_CACHE_CAPACITY, MAX_LENGTH};

/// Errors that can be encountered while using or constructing a `BPE` model.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// An error encountered while reading files mainly.
    #[error("IoError: {0}")]
    Io(#[from] std::io::Error),
    /// When the vocab.json file is in the wrong format
    #[error("Bad vocabulary json file")]
    BadVocabulary,
    /// When the merges.txt file is in the wrong format. This error holds the line
    /// number of the line that caused the error.
    #[error("Merges text file invalid at line {0}")]
    BadMerges(usize),
    /// If a token found in merges, is not in the vocab
    #[error("Token `{0}` out of vocabulary")]
    MergeTokenOutOfVocabulary(String),
    /// If the provided unk token is out of vocabulary
    #[error("Unk token `{0}` not found in the vocabulary")]
    UnkTokenOutOfVocabulary(String),
    /// Dropout not between 0 and 1.
    #[error("Dropout should be between 0 and 1, inclusive")]
    InvalidDropout,
    /// When byte_fallback is enabled but the fallback code is not in the vocab
    #[error("Byte fallback `<{0:#04X}>` not found in the vocabulary")]
    ByteFallbackOutOfVocabulary(u8),
    /// When BPE operating with byte_level the byte atom is not in the vocab
    #[error("Byte atom `{0:#04X}` not found in the vocabulary")]
    ByteAtomOutOfVocabulary(u8),
}

// Re-export
pub use model::*;
pub use serialization::BpeConfig;
