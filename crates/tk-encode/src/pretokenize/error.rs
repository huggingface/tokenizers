use thiserror::Error;
use std::result;

#[derive(Debug, Error)]
pub enum PreTokenizerError {
    #[error("PreTokenizer error")]
    NormalizerError,
}

pub type Result<T> = result::Result<T, PreTokenizerError>;
