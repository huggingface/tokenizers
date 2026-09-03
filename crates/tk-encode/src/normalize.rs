use thiserror::Error;

#[#[derive(Debug, Error)]
pub enum NormalizerError {
    #[error("string too short: {0}")]
    StringTooShort(String),
}
