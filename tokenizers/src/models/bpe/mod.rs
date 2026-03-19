//! [Byte Pair Encoding](https://www.aclweb.org/anthology/P16-1162/) model.
use rustc_hash::FxHashMap;
use std::{iter, mem};

mod model;
mod serialization;
pub mod trainer;
mod word;

type Pair = (u32, u32);

/// Packs a `(u32, u32)` pair into a single `u64` for faster hashing.
#[inline]
fn pack_pair(pair: &Pair) -> u64 {
    (pair.0 as u64) << 32 | pair.1 as u64
}

/// Unpacks a `u64` back into a `(u32, u32)` pair.
#[inline]
fn unpack_pair(packed: u64) -> Pair {
    ((packed >> 32) as u32, packed as u32)
}

/// A merge-lookup map that packs `(u32, u32)` pair keys into single `u64` values
/// for faster hashing (single FxHash multiply instead of hashing two fields).
///
/// Values are `(rank, new_id)` tuples.
#[derive(Clone, Debug)]
pub(crate) struct MergeMap {
    inner: FxHashMap<u64, (u32, u32)>,
}

impl MergeMap {
    #[allow(dead_code)]
    pub fn new() -> Self {
        MergeMap {
            inner: FxHashMap::default(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        MergeMap {
            inner: FxHashMap::with_capacity_and_hasher(cap, Default::default()),
        }
    }

    #[inline]
    pub fn get(&self, pair: &Pair) -> Option<&(u32, u32)> {
        self.inner.get(&pack_pair(pair))
    }

    pub fn insert(&mut self, pair: Pair, value: (u32, u32)) -> Option<(u32, u32)> {
        self.inner.insert(pack_pair(&pair), value)
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Iterate over `(Pair, &(rank, new_id))`.
    pub fn iter(&self) -> impl Iterator<Item = (Pair, &(u32, u32))> {
        self.inner.iter().map(|(k, v)| (unpack_pair(*k), v))
    }
}

impl PartialEq for MergeMap {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl std::iter::FromIterator<(Pair, (u32, u32))> for MergeMap {
    fn from_iter<I: IntoIterator<Item = (Pair, (u32, u32))>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lo, _) = iter.size_hint();
        let mut map = MergeMap::with_capacity(lo);
        for (pair, val) in iter {
            map.insert(pair, val);
        }
        map
    }
}

/// Errors that can be encountered while using or constructing a `BPE` model.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// An error encountered while reading files mainly.
    #[error("IoError: {0}")]
    Io(#[from] std::io::Error),
    /// An error forwarded from Serde, while parsing JSON
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
}

/// Provides access to the `FirstLastIterator` to any Iterator
pub(crate) trait WithFirstLastIterator: Iterator + Sized {
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
pub(crate) struct FirstLastIterator<I>
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
pub use model::*;
pub use trainer::*;
use word::*;
