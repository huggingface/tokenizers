use thiserror::Error;
use std::result;

#[derive(Debug, Error)]
pub enum NormalizerError {
    #[error("Normalizer error")]
    NormalizerError,
}

pub type Result<T> = result::Result<T, NormalizerError>;
