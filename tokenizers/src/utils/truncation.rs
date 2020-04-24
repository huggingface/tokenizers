use crate::tokenizer::{Encoding, Result};

#[derive(Debug, Clone)]
pub struct TruncationParams {
    pub max_length: usize,
    pub strategy: TruncationStrategy,
    pub stride: usize,
}

#[derive(Debug)]
pub enum TruncationError {
    /// We are supposed to truncate the pair sequence, but it has not been provided.
    SecondSequenceNotProvided,
    /// We cannot truncate the target sequence enough to respect the provided max length.
    SequenceTooShort,
    /// We cannot truncate with the given constraints.
    MaxLengthTooLow,
}

impl std::fmt::Display for TruncationError {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        use TruncationError::*;
        match self {
            SecondSequenceNotProvided => {
                write!(fmt, "Truncation error: Second sequence not provided")
            }
            SequenceTooShort => write!(
                fmt,
                "Truncation error: Sequence to truncate too short to respect the provided max_length"
            ),
            MaxLengthTooLow => write!(
                fmt,
                "Truncation error: Specified max length is too low \
                    to respect the various constraints"),
        }
    }
}
impl std::error::Error for TruncationError {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TruncationStrategy {
    LongestFirst,
    OnlyFirst,
    OnlySecond,
}

impl std::convert::AsRef<str> for TruncationStrategy {
    fn as_ref(&self) -> &str {
        match self {
            TruncationStrategy::LongestFirst => "longest_first",
            TruncationStrategy::OnlyFirst => "only_first",
            TruncationStrategy::OnlySecond => "only_second",
        }
    }
}

pub fn truncate_encodings(
    mut encoding: Encoding,
    mut pair_encoding: Option<Encoding>,
    params: &TruncationParams,
) -> Result<(Encoding, Option<Encoding>)> {
    if params.max_length == 0 {
        return Err(Box::new(TruncationError::MaxLengthTooLow));
    }

    let total_length = encoding.get_ids().len()
        + pair_encoding
            .as_ref()
            .map(|e| e.get_ids().len())
            .unwrap_or(0);
    let to_remove = if total_length > params.max_length {
        total_length - params.max_length
    } else {
        return Ok((encoding, pair_encoding));
    };

    match params.strategy {
        TruncationStrategy::LongestFirst => {
            let mut n_first = encoding.get_ids().len();
            let mut n_second = pair_encoding.as_ref().map_or(0, |e| e.get_ids().len());
            for _ in 0..to_remove {
                if n_first > n_second {
                    n_first -= 1;
                } else {
                    n_second -= 1;
                }
            }

            if n_first == 0 || (pair_encoding.is_some() && n_second == 0) {
                return Err(Box::new(TruncationError::MaxLengthTooLow));
            }

            encoding.truncate(n_first, params.stride);
            if let Some(encoding) = pair_encoding.as_mut() {
                encoding.truncate(n_second, params.stride);
            }
        }
        TruncationStrategy::OnlyFirst | TruncationStrategy::OnlySecond => {
            let target = if params.strategy == TruncationStrategy::OnlyFirst {
                Ok(&mut encoding)
            } else if let Some(encoding) = pair_encoding.as_mut() {
                Ok(encoding)
            } else {
                Err(Box::new(TruncationError::SecondSequenceNotProvided))
            }?;

            let target_len = target.get_ids().len();
            if target_len > to_remove {
                target.truncate(target_len - to_remove, params.stride);
            } else {
                return Err(Box::new(TruncationError::SequenceTooShort));
            }
        }
    }

    Ok((encoding, pair_encoding))
}
