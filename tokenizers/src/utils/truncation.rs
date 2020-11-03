use crate::tokenizer::{Encoding, Result};
use serde::{Deserialize, Serialize};
use std::cmp;
use std::mem;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruncationParams {
    pub max_length: usize,
    pub strategy: TruncationStrategy,
    pub stride: usize,
}

impl Default for TruncationParams {
    fn default() -> Self {
        Self {
            max_length: 512,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
        }
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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
        encoding.truncate(0, params.stride);
        if let Some(other_encoding) = pair_encoding.as_mut() {
            other_encoding.truncate(0, params.stride);
        }
        return Ok((encoding, pair_encoding));
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
            if let Some(other_encoding) = pair_encoding.as_mut() {
                // Assuming n1 <= n2, there are 3 cases
                // Case 1:
                //   No truncation needs to be performed.
                //   This scenario is handled before the match.
                // Case 2:
                //   Only the longer input needs to be truncated.
                //   n1 = n1
                //   n2 = max_length - n1
                // Case 3:
                //   Both inputs must be truncated.
                //   n1 = max_length / 2
                //   n2 = n1 + max_length % 2

                let mut n1 = encoding.get_ids().len();
                let mut n2 = other_encoding.get_ids().len();
                let mut swap = false;

                // Ensure n1 is the length of the shortest input
                if n1 > n2 {
                    swap = true;
                    mem::swap(&mut n1, &mut n2);
                }

                if n1 > params.max_length {
                    // This needs to be a special case
                    // to avoid max_length - n1 < 0
                    // since n1 and n2 are unsigned
                    n2 = n1;
                } else {
                    n2 = cmp::max(n1, params.max_length - n1);
                }

                if n1 + n2 > params.max_length {
                    n1 = params.max_length / 2;
                    n2 = n1 + params.max_length % 2;
                }

                // Swap lengths if we swapped previosuly
                if swap {
                    mem::swap(&mut n1, &mut n2);
                }
                encoding.truncate(n1, params.stride);
                other_encoding.truncate(n2, params.stride);
            } else {
                encoding.truncate(total_length - to_remove, params.stride);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::Encoding;
    use std::collections::HashMap;

    fn get_empty() -> Encoding {
        Encoding::new(
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            HashMap::new(),
        )
    }

    fn get_short() -> Encoding {
        Encoding::new(
            vec![1, 2],
            vec![0, 0],
            vec![String::from("a"), String::from("b")],
            vec![Some(0), Some(1)],
            vec![(0, 1), (1, 2)],
            vec![0, 0],
            vec![1, 1],
            vec![],
            HashMap::new(),
        )
    }

    fn get_medium() -> Encoding {
        Encoding::new(
            vec![3, 4, 5, 6],
            vec![0, 0, 0, 0],
            vec![
                String::from("d"),
                String::from("e"),
                String::from("f"),
                String::from("g"),
            ],
            vec![Some(0), Some(1), Some(2), Some(3)],
            vec![(0, 1), (1, 2), (2, 3), (3, 4)],
            vec![0, 0, 0, 0],
            vec![1, 1, 1, 1],
            vec![],
            HashMap::new(),
        )
    }

    fn get_long() -> Encoding {
        Encoding::new(
            vec![7, 8, 9, 10, 11, 12, 13, 14],
            vec![0, 0, 0, 0, 0, 0, 0, 0],
            vec![
                String::from("h"),
                String::from("i"),
                String::from("j"),
                String::from("k"),
                String::from("l"),
                String::from("m"),
                String::from("n"),
                String::from("o"),
            ],
            vec![
                Some(0),
                Some(1),
                Some(2),
                Some(3),
                Some(4),
                Some(5),
                Some(6),
                Some(7),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (6, 8),
            ],
            vec![0, 0, 0, 0, 0, 0, 0, 0],
            vec![1, 1, 1, 1, 1, 1, 1, 1],
            vec![],
            HashMap::new(),
        )
    }

    fn truncate_and_assert(
        encoding1: Encoding,
        encoding2: Encoding,
        params: &TruncationParams,
        n1: usize,
        n2: usize,
    ) {
        match truncate_encodings(encoding1, Some(encoding2), &params) {
            Ok((e1, Some(e2))) => {
                assert!(e1.get_ids().len() == n1);
                assert!(e2.get_ids().len() == n2);
            }
            _ => panic!(),
        };
    }

    #[test]
    fn truncate_encodings_longest_first() {
        let params = TruncationParams {
            max_length: 7,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
        };

        truncate_and_assert(get_empty(), get_empty(), &params, 0, 0);
        truncate_and_assert(get_empty(), get_short(), &params, 0, 2);
        truncate_and_assert(get_empty(), get_medium(), &params, 0, 4);
        truncate_and_assert(get_empty(), get_long(), &params, 0, 7);

        truncate_and_assert(get_short(), get_empty(), &params, 2, 0);
        truncate_and_assert(get_short(), get_short(), &params, 2, 2);
        truncate_and_assert(get_short(), get_medium(), &params, 2, 4);
        truncate_and_assert(get_short(), get_long(), &params, 2, 5);

        truncate_and_assert(get_medium(), get_empty(), &params, 4, 0);
        truncate_and_assert(get_medium(), get_short(), &params, 4, 2);
        truncate_and_assert(get_medium(), get_medium(), &params, 3, 4);
        truncate_and_assert(get_medium(), get_long(), &params, 3, 4);

        truncate_and_assert(get_long(), get_empty(), &params, 7, 0);
        truncate_and_assert(get_long(), get_short(), &params, 5, 2);
        truncate_and_assert(get_long(), get_medium(), &params, 4, 3);
        truncate_and_assert(get_long(), get_long(), &params, 3, 4);
    }

    #[test]
    fn truncate_encodings_empty() {
        let params = TruncationParams {
            max_length: 0,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
        };

        truncate_and_assert(get_empty(), get_short(), &params, 0, 0);
        truncate_and_assert(get_medium(), get_medium(), &params, 0, 0);
        truncate_and_assert(get_long(), get_long(), &params, 0, 0);
    }
}
