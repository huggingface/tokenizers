use crate::tokenizer::{Encoding, PaddingDirection, Result};

#[derive(Debug, Clone)]
pub struct TruncationParams {
    pub max_length: usize,
    pub strategy: TruncationStrategy,
    pub stride: usize,
}

#[derive(Debug, Clone)]
pub struct PaddingParams {
    pub strategy: PaddingStrategy,
    pub direction: PaddingDirection,
    pub pad_id: u32,
    pub pad_type_id: u32,
    pub pad_token: String,
}

#[derive(Debug, Clone)]
pub enum PaddingStrategy {
    BatchLongest,
    Fixed(usize),
}

#[derive(Debug)]
pub enum Error {
    SecondSequenceNotProvided,
}

impl std::fmt::Display for Error {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::SecondSequenceNotProvided => {
                write!(fmt, "Truncation error: Second sequence not provided")
            }
        }
    }
}
impl std::error::Error for Error {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TruncationStrategy {
    LongestFirst,
    OnlyFirst,
    OnlySecond,
}

pub fn truncate_encodings(
    mut encoding: Encoding,
    mut pair_encoding: Option<Encoding>,
    params: &TruncationParams,
) -> Result<(Encoding, Option<Encoding>)> {
    if params.max_length == 0 {
        return Ok((encoding, pair_encoding));
    }

    match params.strategy {
        TruncationStrategy::LongestFirst => {
            let total_length = encoding.get_ids().len()
                + pair_encoding
                    .as_ref()
                    .map(|e| e.get_ids().len())
                    .unwrap_or(0);
            let to_remove = if total_length > params.max_length {
                total_length - params.max_length
            } else {
                0
            };

            let mut n_first = 0;
            let mut n_second = 0;
            for _ in 0..to_remove {
                if pair_encoding.is_none()
                    || encoding.get_ids().len() > pair_encoding.as_ref().unwrap().get_ids().len()
                {
                    n_first += 1;
                } else {
                    n_second += 1;
                }
            }

            encoding.truncate(encoding.get_ids().len() - n_first, params.stride);
            if let Some(encoding) = pair_encoding.as_mut() {
                encoding.truncate(encoding.get_ids().len() - n_second, params.stride);
            }
        }
        TruncationStrategy::OnlyFirst | TruncationStrategy::OnlySecond => {
            let target = if params.strategy == TruncationStrategy::OnlyFirst {
                Ok(&mut encoding)
            } else if let Some(encoding) = pair_encoding.as_mut() {
                Ok(encoding)
            } else {
                Err(Box::new(Error::SecondSequenceNotProvided))
            }?;

            if target.get_ids().len() > params.max_length {
                target.truncate(params.max_length, params.stride);
            }
        }
    }

    Ok((encoding, pair_encoding))
}

pub fn pad_encodings(
    mut encodings: Vec<Encoding>,
    params: &PaddingParams,
) -> Result<Vec<Encoding>> {
    if encodings.is_empty() {
        return Ok(encodings);
    }

    let pad_length = match params.strategy {
        PaddingStrategy::Fixed(size) => size,
        PaddingStrategy::BatchLongest => encodings.iter().map(|e| e.get_ids().len()).max().unwrap(),
    };

    for encoding in encodings.iter_mut() {
        encoding.pad(
            pad_length,
            params.pad_id,
            params.pad_type_id,
            &params.pad_token,
            &params.direction,
        );
    }

    Ok(encodings)
}
