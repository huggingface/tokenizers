use std::{convert::From, io};

mod cache;
mod model;
mod trainer;
mod word;

pub type Pair = (u32, u32);

/// ## Error
/// Errors that can be encountered while using BPE
#[derive(Debug)]
pub enum Error {
    /// An error encountered while reading files mainly.
    Io(std::io::Error),
    /// An error forwarded from Serde, while parsing JSON
    JsonError(serde_json::Error),
    /// When the vocab.json file is in the wrong format
    BadVocabulary,
    /// If a token found in merges, is not in the vocab
    MergeTokenOutOfVocabulary(String),
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

// Re-export
pub use cache::*;
pub use model::*;
pub use trainer::*;
pub use word::*;
