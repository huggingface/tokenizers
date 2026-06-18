use thiserror::Error;

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("Normalizer error: {0}")]
    Normalizer(NormalizerError),
}

pub type Result<T> = std::result::Result<T, TokenizerError>;
