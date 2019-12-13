use crate::tokenizer::{Encoding, Result};

#[derive(Debug)]
pub enum Error {
    SequenceTooSmall,
    SecondSequenceNotProvided,
}

impl std::fmt::Display for Error {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::SequenceTooSmall => write!(fmt, "Truncation error: Sequence is too small"),
            Error::SecondSequenceNotProvided => {
                write!(fmt, "Truncation error: Second sequence not provided")
            }
        }
    }
}
impl std::error::Error for Error {}

#[derive(Clone, Copy, PartialEq)]
pub enum TruncationStrategy {
    LongestFirst,
    OnlyFirst,
    OnlySecond,
}

pub fn truncate_encodings(
    mut encoding: Encoding,
    mut pair_encoding: Option<Encoding>,
    to_remove: usize,
    strategy: TruncationStrategy,
    stride: usize,
) -> Result<(Encoding, Option<Encoding>)> {
    if to_remove == 0 {
        return Ok((encoding, pair_encoding));
    }

    match strategy {
        TruncationStrategy::LongestFirst => {
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

            encoding.truncate(encoding.get_ids().len() - n_first, stride);
            if let Some(encoding) = pair_encoding.as_mut() {
                encoding.truncate(encoding.get_ids().len() - n_second, stride);
            }
        }
        TruncationStrategy::OnlyFirst | TruncationStrategy::OnlySecond => {
            let target = if strategy == TruncationStrategy::OnlyFirst {
                Ok(&mut encoding)
            } else if let Some(encoding) = pair_encoding.as_mut() {
                Ok(encoding)
            } else {
                Err(Box::new(Error::SecondSequenceNotProvided))
            }?;

            if target.get_ids().len() <= to_remove {
                return Err(Box::new(Error::SequenceTooSmall));
            }

            target.truncate(target.get_ids().len() - to_remove, stride);
        }
    }

    Ok((encoding, pair_encoding))
}
