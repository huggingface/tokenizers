//! [Byte n-gram Encoding]
use std::cmp::Ordering;
use std::fmt;
use std::{iter, mem};

mod model;
mod serialization;
pub mod trainer;
mod word;

//TODO: Change to n-gram (maybe, pair of start + length)
#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Debug, Eq, Hash, Clone)]
pub struct Ngram {
    ids: Vec<u32>,
}
impl PartialEq for Ngram {
    fn eq(&self, other: &Self) -> bool {
        if self.ids.len() != other.ids.len() {
            return false;
        }
        for i in 0..self.ids.len() {
            if self.ids[i] != other.ids[i] {
                return false;
            }
        }
        true
    }
}
impl PartialOrd for Ngram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Ngram {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.ids.len() != other.ids.len() {
            self.ids.len().cmp(&other.ids.len())
        } else {
            self.ids
                .iter()
                .zip(other.ids.clone())
                .map(|(x, y)| x.cmp(&y))
                .find(|&o| o != Ordering::Equal)
                .unwrap_or(Ordering::Equal)
        }
    }
}
// For testing purposes
impl fmt::Display for Ngram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ngram = self.clone();
        write!(
            f,
            "Ngram: ids[{}]",
            ngram
                .ids
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}
/// Errors that can be encountered while using or constructing a `BNE` model.
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

/*
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ngram_fmt() {
        let a = Ngram {
            ids: vec![1, 2, 5, 3, 5]
        };
        assert_eq!("Ngram: ids[1, 2, 5, 3, 5]", a.to_string());
    }
}
*/
