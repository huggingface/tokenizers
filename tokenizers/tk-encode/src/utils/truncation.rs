use crate::pipeline::PipelineToken;
use crate::tokenizer::Result;
use std::cmp;
use std::mem;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TruncationDirection {
    Left,
    #[default]
    Right,
}

impl std::convert::AsRef<str> for TruncationDirection {
    fn as_ref(&self) -> &str {
        match self {
            TruncationDirection::Left => "left",
            TruncationDirection::Right => "right",
        }
    }
}

#[derive(Debug, Clone)]
pub struct TruncationParams {
    pub direction: TruncationDirection,
    pub max_length: usize,
    pub strategy: TruncationStrategy,
    pub stride: usize,
}

impl Default for TruncationParams {
    fn default() -> Self {
        Self {
            max_length: 512,
            strategy: TruncationStrategy::default(),
            stride: 0,
            direction: TruncationDirection::default(),
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum TruncationError {
    /// We are supposed to truncate the pair sequence, but it has not been provided.
    #[error("Truncation error: Second sequence not provided")]
    SecondSequenceNotProvided,
    /// We cannot truncate the target sequence enough to respect the provided max length.
    #[error("Truncation error: Sequence to truncate too short to respect the provided max_length")]
    SequenceTooShort,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TruncationStrategy {
    #[default]
    LongestFirst,
    OnlyFirst,
    OnlySecond,
}

impl std::convert::AsRef<str> for TruncationStrategy {
    fn as_ref(&self) -> &str {
        match self {
            Self::LongestFirst => "longest_first",
            Self::OnlyFirst => "only_first",
            Self::OnlySecond => "only_second",
        }
    }
}

pub fn pipeline_truncate_pair(
    mut s1: Vec<PipelineToken>,
    maybe_s2: Option<Vec<PipelineToken>>,
    truncation: &Option<TruncationParams>,
    num_special_tokens: usize,
) -> Result<(Vec<PipelineToken>, Option<Vec<PipelineToken>>)> {
    let seq_len = s1.len() + maybe_s2.as_ref().map_or(0, Vec::len);
    let total_len = num_special_tokens + seq_len;

    if let Some(truncation) = truncation {
        if truncation.max_length == 0 {
            // XXX: determine whether we should return an empty Encoding or just the specials here
            // TODO: better wording for the warn log
            warn!("Truncation `max_length` was set to 0: returning an empty sequence");
            Ok((Vec::new(), maybe_s2.and(Some(Vec::new()))))
        } else if total_len > truncation.max_length {
            let num_removed = total_len - truncation.max_length;

            match truncation.strategy {
                TruncationStrategy::LongestFirst => {
                    if let Some(mut s2) = maybe_s2 {
                        // XXX: this algorithm is ported from the legacy truncation code, verbatim
                        // Could probably be rewritten to be simpler / clearer
                        let mut n1 = s1.len();
                        let mut n2 = s2.len();
                        let mut swap = false;

                        // Ensure n1 is the length of the shortest input
                        if n1 > n2 {
                            swap = true;
                            mem::swap(&mut n1, &mut n2);
                        }

                        if n1 > truncation.max_length {
                            // This needs to be a special case
                            // to avoid max_length - n1 < 0
                            // since n1 and n2 are unsigned
                            n2 = n1;
                        } else {
                            n2 = cmp::max(n1, truncation.max_length - n1);
                        }

                        if n1 + n2 > truncation.max_length {
                            n1 = truncation.max_length / 2;
                            n2 = n1 + truncation.max_length % 2;
                        }

                        // Swap lengths if we swapped previously
                        if swap {
                            mem::swap(&mut n1, &mut n2);
                        }
                        s1.truncate(n1);
                        s2.truncate(n2);
                        Ok((s1, Some(s2)))
                    } else {
                        s1.truncate(s1.len() - num_removed);
                        Ok((s1, maybe_s2))
                    }
                }
                TruncationStrategy::OnlyFirst | TruncationStrategy::OnlySecond => {
                    let (mut sequence_to_truncate, other) =
                        if truncation.strategy == TruncationStrategy::OnlyFirst {
                            (s1, maybe_s2)
                        } else if maybe_s2.is_some() {
                            (maybe_s2.unwrap(), Some(s1))
                        } else {
                            return Err(Box::new(TruncationError::SecondSequenceNotProvided));
                        };
                    let sequence_length = sequence_to_truncate.len();
                    if sequence_length > num_removed {
                        sequence_to_truncate.truncate(sequence_length - num_removed);
                    } else {
                        return Err(Box::new(TruncationError::SequenceTooShort));
                    }
                    if truncation.strategy == TruncationStrategy::OnlyFirst {
                        Ok((sequence_to_truncate, other))
                    } else {
                        Ok((other.unwrap(), Some(sequence_to_truncate)))
                    }
                }
            }
        } else {
            // No need to truncate
            Ok((s1, maybe_s2))
        }
    } else {
        // None config = no truncation
        Ok((s1, maybe_s2))
    }
}

// pub fn truncate_encodings(
//     mut encoding: Encoding,
//     mut pair_encoding: Option<Encoding>,
//     params: &TruncationParams,
// ) -> Result<(Encoding, Option<Encoding>)> {
//     if params.max_length == 0 {
//         encoding.truncate(0, params.stride, params.direction);
//         if let Some(other_encoding) = pair_encoding.as_mut() {
//             other_encoding.truncate(0, params.stride, params.direction);
//         }
//         return Ok((encoding, pair_encoding));
//     }

//     let total_length = encoding.get_ids().len()
//         + pair_encoding
//             .as_ref()
//             .map(|e| e.get_ids().len())
//             .unwrap_or(0);
//     let to_remove = if total_length > params.max_length {
//         total_length - params.max_length
//     } else {
//         return Ok((encoding, pair_encoding));
//     };

//     match params.strategy {
//         TruncationStrategy::LongestFirst => {
//             if let Some(other_encoding) = pair_encoding.as_mut() {
//                 // Assuming n1 <= n2, there are 3 cases
//                 // Case 1:
//                 //   No truncation needs to be performed.
//                 //   This scenario is handled before the match.
//                 // Case 2:
//                 //   Only the longer input needs to be truncated.
//                 //   n1 = n1
//                 //   n2 = max_length - n1
//                 // Case 3:
//                 //   Both inputs must be truncated.
//                 //   n1 = max_length / 2
//                 //   n2 = n1 + max_length % 2

//                 let mut n1 = encoding.get_ids().len();
//                 let mut n2 = other_encoding.get_ids().len();
//                 let mut swap = false;

//                 // Ensure n1 is the length of the shortest input
//                 if n1 > n2 {
//                     swap = true;
//                     mem::swap(&mut n1, &mut n2);
//                 }

//                 if n1 > params.max_length {
//                     // This needs to be a special case
//                     // to avoid max_length - n1 < 0
//                     // since n1 and n2 are unsigned
//                     n2 = n1;
//                 } else {
//                     n2 = cmp::max(n1, params.max_length - n1);
//                 }

//                 if n1 + n2 > params.max_length {
//                     n1 = params.max_length / 2;
//                     n2 = n1 + params.max_length % 2;
//                 }

//                 // Swap lengths if we swapped previously
//                 if swap {
//                     mem::swap(&mut n1, &mut n2);
//                 }
//                 encoding.truncate(n1, params.stride, params.direction);
//                 other_encoding.truncate(n2, params.stride, params.direction);
//             } else {
//                 encoding.truncate(total_length - to_remove, params.stride, params.direction);
//             }
//         }
//         TruncationStrategy::OnlyFirst | TruncationStrategy::OnlySecond => {
//             let target = if params.strategy == TruncationStrategy::OnlyFirst {
//                 Ok(&mut encoding)
//             } else if let Some(encoding) = pair_encoding.as_mut() {
//                 Ok(encoding)
//             } else {
//                 Err(Box::new(TruncationError::SecondSequenceNotProvided))
//             }?;

//             let target_len = target.get_ids().len();
//             if target_len > to_remove {
//                 target.truncate(target_len - to_remove, params.stride, params.direction);
//             } else {
//                 return Err(Box::new(TruncationError::SequenceTooShort));
//             }
//         }
//     }
//     Ok((encoding, pair_encoding))
// }

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tokens(ids: impl IntoIterator<Item = u32>) -> Vec<PipelineToken> {
        ids.into_iter().map(PipelineToken::from).collect()
    }

    fn empty() -> Vec<PipelineToken> {
        Vec::new()
    }

    fn short() -> Vec<PipelineToken> {
        make_tokens(1..3)
    }

    fn medium() -> Vec<PipelineToken> {
        make_tokens(3..7)
    }

    fn long() -> Vec<PipelineToken> {
        make_tokens(7..15)
    }

    fn params(max_length: usize, strategy: TruncationStrategy) -> Option<TruncationParams> {
        Some(TruncationParams {
            max_length,
            strategy,
            ..TruncationParams::default()
        })
    }

    fn truncate_and_assert(
        s1: Vec<PipelineToken>,
        s2: Vec<PipelineToken>,
        truncation: &Option<TruncationParams>,
        n1: usize,
        n2: usize,
    ) {
        let (t1, t2) = pipeline_truncate_pair(s1, Some(s2), truncation, 0).unwrap();
        assert_eq!(t1.len(), n1);
        assert_eq!(t2.expect("the pair is kept").len(), n2);
    }

    #[test]
    fn longest_first_balances_a_pair_against_max_length() {
        let params = params(7, TruncationStrategy::LongestFirst);

        truncate_and_assert(empty(), empty(), &params, 0, 0);
        truncate_and_assert(empty(), short(), &params, 0, 2);
        truncate_and_assert(empty(), medium(), &params, 0, 4);
        truncate_and_assert(empty(), long(), &params, 0, 7);

        truncate_and_assert(short(), empty(), &params, 2, 0);
        truncate_and_assert(short(), short(), &params, 2, 2);
        truncate_and_assert(short(), medium(), &params, 2, 4);
        truncate_and_assert(short(), long(), &params, 2, 5);

        truncate_and_assert(medium(), empty(), &params, 4, 0);
        truncate_and_assert(medium(), short(), &params, 4, 2);
        truncate_and_assert(medium(), medium(), &params, 3, 4);
        truncate_and_assert(medium(), long(), &params, 3, 4);

        truncate_and_assert(long(), empty(), &params, 7, 0);
        truncate_and_assert(long(), short(), &params, 5, 2);
        truncate_and_assert(long(), medium(), &params, 4, 3);
        truncate_and_assert(long(), long(), &params, 3, 4);
    }

    #[test]
    fn no_truncation_params_keeps_both_sequences_whole() {
        truncate_and_assert(long(), long(), &None, 8, 8);
    }

    #[test]
    fn longest_first_drops_the_tail_of_a_lone_sequence() {
        let (t1, t2) = pipeline_truncate_pair(
            long(),
            None,
            &params(3, TruncationStrategy::LongestFirst),
            0,
        )
        .unwrap();

        assert_eq!(t1, make_tokens(7..10));
        assert!(t2.is_none());
    }

    // The specials the post-processor will add are not in either sequence yet, so the caller passes
    // their count and truncation has to make room for them.
    #[test]
    fn special_tokens_count_against_max_length() {
        let params = params(8, TruncationStrategy::LongestFirst);

        let (untouched, _) = pipeline_truncate_pair(long(), None, &params, 0).unwrap();
        assert_eq!(untouched, long());

        let (shortened, _) = pipeline_truncate_pair(long(), None, &params, 2).unwrap();
        assert_eq!(shortened, make_tokens(7..13));
    }

    #[test]
    fn only_first_truncates_the_first_sequence() {
        let (t1, t2) = pipeline_truncate_pair(
            long(),
            Some(short()),
            &params(7, TruncationStrategy::OnlyFirst),
            0,
        )
        .unwrap();

        assert_eq!(t1, make_tokens(7..12));
        assert_eq!(t2, Some(short()));
    }

    #[test]
    fn only_second_truncates_the_second_sequence() {
        let (t1, t2) = pipeline_truncate_pair(
            short(),
            Some(long()),
            &params(7, TruncationStrategy::OnlySecond),
            0,
        )
        .unwrap();

        assert_eq!(t1, short());
        assert_eq!(t2, Some(make_tokens(7..12)));
    }

    // `OnlySecond` names the sequence to cut, so there is nothing to cut without a pair. Silently
    // falling back to the first sequence would truncate what the caller asked us to keep.
    #[test]
    fn only_second_refuses_a_missing_pair() {
        let err =
            pipeline_truncate_pair(long(), None, &params(7, TruncationStrategy::OnlySecond), 0)
                .err()
                .unwrap();

        assert!(matches!(
            err.downcast_ref::<TruncationError>(),
            Some(TruncationError::SecondSequenceNotProvided)
        ));
    }

    // `OnlyFirst` forbids touching the pair, so a first sequence that is already shorter than what
    // has to go cannot reach `max_length` at all.
    #[test]
    fn only_first_refuses_a_sequence_too_short_to_truncate() {
        let err = pipeline_truncate_pair(
            short(),
            Some(long()),
            &params(1, TruncationStrategy::OnlyFirst),
            0,
        )
        .err()
        .unwrap();

        assert!(matches!(
            err.downcast_ref::<TruncationError>(),
            Some(TruncationError::SequenceTooShort)
        ));
    }

    // `max_length` 0 empties both sequences, but the pair still has to come back as `Some`: the
    // post-processor's pair template references the second sequence, and it cannot place the
    // specials around a sequence that is not there.
    #[test]
    fn max_length_zero_empties_both_sequences() {
        let params = params(0, TruncationStrategy::LongestFirst);

        truncate_and_assert(empty(), short(), &params, 0, 0);
        truncate_and_assert(medium(), medium(), &params, 0, 0);
        truncate_and_assert(long(), long(), &params, 0, 0);
    }
}
