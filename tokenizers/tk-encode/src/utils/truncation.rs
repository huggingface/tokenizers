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

pub fn truncate_pair(
    mut s1: Vec<PipelineToken>,
    maybe_s2: Option<Vec<PipelineToken>>,
    truncation: &Option<TruncationParams>,
    num_added_special_tokens: usize,
) -> Result<(Vec<PipelineToken>, Option<Vec<PipelineToken>>)> {
    let seq_len = s1.len() + maybe_s2.as_ref().map_or(0, Vec::len);

    if let Some(truncation) = truncation {
        let truncate_to_length = truncation.max_length.saturating_sub(num_added_special_tokens);

        if truncate_to_length == 0 {
            // XXX: maybe we should error out when instantiating the PipelineTokenizer to avoid this
            warn!(
                "Truncation max_length is too short to include the tokens: `max_length` is {}, the post-processor adds {num_added_special_tokens} special tokens. Returning an empty sequence",
                truncation.max_length
            );
            Ok((Vec::new(), maybe_s2.and(Some(Vec::new()))))
        } else if seq_len > truncate_to_length {
            let num_removed = seq_len - truncate_to_length;

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

                        if n1 > truncate_to_length {
                            // This needs to be a special case
                            // to avoid max_length - n1 < 0
                            // since n1 and n2 are unsigned
                            n2 = n1;
                        } else {
                            n2 = cmp::max(n1, truncate_to_length - n1);
                        }

                        if n1 + n2 > truncate_to_length {
                            n1 = truncate_to_length / 2;
                            n2 = n1 + truncate_to_length % 2;
                        }

                        // Swap lengths if we swapped previously
                        if swap {
                            mem::swap(&mut n1, &mut n2);
                        }
                        truncate_tokens(&mut s1, n1, truncation.direction);
                        truncate_tokens(&mut s2, n2, truncation.direction);
                        Ok((s1, Some(s2)))
                    } else {
                        let len = s1.len();
                        truncate_tokens(&mut s1, len - num_removed, truncation.direction);
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
                        truncate_tokens(
                            &mut sequence_to_truncate,
                            sequence_length - num_removed,
                            truncation.direction,
                        );
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

/// Truncates tokens to `keep` tokens.
fn truncate_tokens(tokens: &mut Vec<PipelineToken>, keep: usize, direction: TruncationDirection) {
    if direction == TruncationDirection::Left {
        tokens.drain(..tokens.len().saturating_sub(keep));
    } else {
        tokens.truncate(keep);
    }
}

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
        let (t1, t2) = truncate_pair(s1, Some(s2), truncation, 0).unwrap();
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
        let (t1, t2) = truncate_pair(
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

        let (untouched, _) = truncate_pair(long(), None, &params, 0).unwrap();
        assert_eq!(untouched, long());

        let (shortened, _) = truncate_pair(long(), None, &params, 2).unwrap();
        assert_eq!(shortened, make_tokens(7..13));
    }

    #[test]
    fn only_first_truncates_the_first_sequence() {
        let (t1, t2) = truncate_pair(
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
        let (t1, t2) = truncate_pair(
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
        let err = truncate_pair(long(), None, &params(7, TruncationStrategy::OnlySecond), 0)
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
        let err = truncate_pair(
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

    const BOTH_DIRECTIONS: [TruncationDirection; 2] =
        [TruncationDirection::Left, TruncationDirection::Right];

    fn truncated(
        mut tokens: Vec<PipelineToken>,
        keep: usize,
        direction: TruncationDirection,
    ) -> Vec<PipelineToken> {
        truncate_tokens(&mut tokens, keep, direction);
        tokens
    }

    #[test]
    fn right_truncation_keeps_the_head() {
        assert_eq!(
            truncated(long(), 3, TruncationDirection::Right),
            make_tokens(7..10)
        );
    }

    #[test]
    fn left_truncation_keeps_the_tail() {
        assert_eq!(
            truncated(long(), 3, TruncationDirection::Left),
            make_tokens(12..15)
        );
    }

    #[test]
    fn truncating_to_the_sequence_length_leaves_it_untouched() {
        for direction in BOTH_DIRECTIONS {
            assert_eq!(truncated(long(), 8, direction), long());
        }
    }

    // The pair strategy can ask to keep more tokens than a sequence holds, which is why the left
    // branch saturates: `num_special_tokens` 10 against a 2 and 5 token pair with `max_length` 9
    // computes 7 tokens to keep out of the 5 the second sequence has.
    #[test]
    fn keeping_more_tokens_than_there_are_leaves_the_sequence_untouched() {
        for direction in BOTH_DIRECTIONS {
            assert_eq!(truncated(medium(), 9, direction), medium());
            assert_eq!(truncated(empty(), 9, direction), empty());
        }
    }

    #[test]
    fn truncating_to_zero_empties_the_sequence() {
        for direction in BOTH_DIRECTIONS {
            assert!(truncated(long(), 0, direction).is_empty());
        }
    }

    // Draining the head keeps the buffer the tokens are already in. Collecting the tail into a new
    // `Vec` would pay an allocation and a copy instead, which is the whole reason the left branch
    // is written as a drain.
    #[test]
    fn left_truncation_reuses_the_allocation() {
        let mut tokens = long();
        let capacity = tokens.capacity();
        let address = tokens.as_ptr();

        truncate_tokens(&mut tokens, 3, TruncationDirection::Left);

        assert_eq!(tokens.capacity(), capacity);
        assert_eq!(tokens.as_ptr(), address);
    }

    fn assert_truncated(
        s1: Vec<PipelineToken>,
        s2: Option<Vec<PipelineToken>>,
        truncation: TruncationParams,
        num_special_tokens: usize,
        expected1: &[u32],
        expected2: Option<&[u32]>,
    ) {
        let (t1, t2) = truncate_pair(s1, s2, &Some(truncation), num_special_tokens).unwrap();

        assert_eq!(t1, make_tokens(expected1.iter().copied()));
        assert_eq!(t2, expected2.map(|ids| make_tokens(ids.iter().copied())));
    }

    fn left(max_length: usize, strategy: TruncationStrategy) -> TruncationParams {
        TruncationParams {
            max_length,
            strategy,
            direction: TruncationDirection::Left,
            ..TruncationParams::default()
        }
    }

    // Every other test here runs the default `Right` direction, so nothing pins that each strategy
    // passes the configured direction on to `truncate_tokens`. Expected ids come from released
    // tokenizers 0.23.1 `truncate_encodings`, which is where the direction semantics come from.
    #[test]
    fn the_left_direction_reaches_every_strategy() {
        assert_truncated(
            long(),
            Some(long()),
            left(7, TruncationStrategy::LongestFirst),
            0,
            &[12, 13, 14],
            Some(&[11, 12, 13, 14]),
        );
        assert_truncated(
            long(),
            None,
            left(3, TruncationStrategy::LongestFirst),
            0,
            &[12, 13, 14],
            None,
        );
        assert_truncated(
            long(),
            Some(short()),
            left(7, TruncationStrategy::OnlyFirst),
            0,
            &[10, 11, 12, 13, 14],
            Some(&[1, 2]),
        );
        assert_truncated(
            short(),
            Some(long()),
            left(7, TruncationStrategy::OnlySecond),
            0,
            &[1, 2],
            Some(&[10, 11, 12, 13, 14]),
        );
    }

    // Released tokenizers 0.23.1 subtracts the specials from `max_length` before balancing a pair,
    // so 4 and 4 tokens with 3 specials and `max_length` 8 come back as 2 and 3. Expected ids are
    // its output.
    #[test]
    fn specials_count_against_max_length_for_a_pair() {
        assert_truncated(
            medium(),
            Some(medium()),
            TruncationParams {
                max_length: 8,
                strategy: TruncationStrategy::LongestFirst,
                ..TruncationParams::default()
            },
            3,
            &[3, 4],
            Some(&[3, 4, 5]),
        );
    }
}
