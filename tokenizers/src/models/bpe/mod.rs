//! [Byte Pair Encoding](https://www.aclweb.org/anthology/P16-1162/) model.
use std::{convert::From, io, iter, mem};

mod model;
mod serialization;
mod trainer;
mod word;

type Pair = (u32, u32);

/// Errors that can be encountered while using or constructing a `BPE` model.
#[derive(Debug)]
pub enum Error {
    /// An error encountered while reading files mainly.
    Io(std::io::Error),
    /// An error forwarded from Serde, while parsing JSON
    JsonError(serde_json::Error),
    /// When the vocab.json file is in the wrong format
    BadVocabulary,
    /// When the merges.txt file is in the wrong format. This error holds the line
    /// number of the line that caused the error.
    BadMerges(usize),
    /// If a token found in merges, is not in the vocab
    MergeTokenOutOfVocabulary(String),
    /// If the provided unk token is out of vocabulary
    UnkTokenOutOfVocabulary(String),
    /// Dropout not between 0 and 1.
    InvalidDropout,
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Error::Io(error)
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Error::JsonError(error)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "IoError: {}", e),
            Error::JsonError(e) => write!(f, "JsonError: {}", e),
            Error::BadVocabulary => write!(f, "Bad vocabulary json file"),
            Error::BadMerges(line) => write!(f, "Merges text file invalid at line {}", line),
            Error::MergeTokenOutOfVocabulary(token) => {
                write!(f, "Token `{}` out of vocabulary", token)
            }
            Error::UnkTokenOutOfVocabulary(token) => {
                write!(f, "Unk token `{}` not found in the vocabulary", token)
            }
            Error::InvalidDropout => write!(f, "Dropout should be between 0 and 1"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::JsonError(e) => Some(e),
            _ => None,
        }
    }
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
