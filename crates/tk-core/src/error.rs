use thiserror::Error;

#[derive(Debug, Error)]
pub enum TokenizerError {
	#[error("normalization error")]
	NormalizerError
}

