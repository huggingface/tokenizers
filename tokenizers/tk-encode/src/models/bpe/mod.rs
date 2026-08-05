//! [Byte Pair Encoding](https://www.aclweb.org/anthology/P16-1162/) model.
use std::{iter, mem};
mod bpe_build_tables;
mod bpe_model;
mod bpe_pretoken_to_rank;
mod bpe_scratch;
mod bytelevel_folding;
mod legacy_model;
#[cfg(feature = "config")]
mod legacy_serialization;
pub mod legacy_word;
mod merge_hot_cold_queue;
mod merge_multipass;

#[cfg(test)]
mod tests;

pub type Pair = (u32, u32);

/// Errors that can be encountered while using or constructing a `BPE` model.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// An error encountered while reading files mainly.
    #[error("IoError: {0}")]
    Io(#[from] std::io::Error),
    /// An error forwarded from Serde, while parsing JSON. Behind `config`: without it nothing
    /// here parses JSON, and the variant alone would keep the whole parser linked.
    #[cfg(feature = "config")]
    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),
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

/// Provides access to the `FirstLastIterator` to any Iterator
pub trait WithFirstLastIterator: Iterator + Sized {
    fn with_first_and_last(self) -> FirstLastIterator<Self>;
}

impl<I> WithFirstLastIterator for I
where
    I: Iterator,
{
    fn with_first_and_last(self) -> FirstLastIterator<Self> {
        FirstLastIterator {
            first: true,
            iter: self.peekable(),
        }
    }
}

/// Provides information about whether an item is the first and/or the last of the iterator
pub struct FirstLastIterator<I>
where
    I: Iterator,
{
    first: bool,
    iter: iter::Peekable<I>,
}

impl<I> Iterator for FirstLastIterator<I>
where
    I: Iterator,
{
    /// (is_first, is_last, item)
    type Item = (bool, bool, I::Item);

    fn next(&mut self) -> Option<Self::Item> {
        let first = mem::replace(&mut self.first, false);
        self.iter
            .next()
            .map(|e| (first, self.iter.peek().is_none(), e))
    }
}

// Re-export
pub use bpe_model::*;
pub use bpe_scratch::*;
pub use legacy_model::*;
pub use legacy_word::*;
