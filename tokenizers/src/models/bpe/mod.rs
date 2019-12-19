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
    /// When the merges.txt file is in the wrong format. This error holds the line
    /// number of the line that caused the error.
    BadMerges(usize),
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

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "IoError: {}", e),
            Error::JsonError(e) => write!(f, "JsonError: {}", e),
            Error::BadVocabulary => write!(f, "Bad vocabulary json file"),
            Error::BadMerges(line) => write!(f, "Merges text file invalid at line {}", line),
            Error::MergeTokenOutOfVocabulary(token) => {
                write!(f, "Token {} out of vocabulary", token)
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::JsonError(e) => Some(e),
            Error::BadVocabulary => None,
            Error::BadMerges(_) => None,
            Error::MergeTokenOutOfVocabulary(_) => None,
        }
    }
}

// Re-export
pub use cache::*;
pub use model::*;
pub use trainer::*;
pub use word::*;
