use crate::pipeline::PipelineToken;
use crate::tokenizer::Result;
use std::cmp;
use std::mem;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
    /// The stride leaves no room for a window to advance. The room is what the sequence actually
    /// windows at, which is `max_length` minus the special tokens for one sequence and less for
    /// a `LongestFirst` pair, where it is shared between the two sides.
    #[error(
        "Truncation error: `stride` ({stride}) must be strictly less than the tokens each window keeps ({room})"
    )]
    StrideTooLarge { stride: usize, room: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
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

/// What a pair looks like after truncation: the tokens that stay in each sequence, and the
/// windows their removed parts re-tile into.
pub struct TruncatedPair {
    pub first: Vec<PipelineToken>,
    /// The windows of the first sequence, in the order the released crate returns them. Empty
    /// when the first sequence was not cut.
    pub first_windows: Vec<Vec<PipelineToken>>,
    pub second: Option<Vec<PipelineToken>>,
    /// The windows of the second sequence. Empty when there is no second sequence or it was not
    /// cut.
    pub second_windows: Vec<Vec<PipelineToken>>,
}

pub fn truncate_pair(
    s1: Vec<PipelineToken>,
    maybe_s2: Option<Vec<PipelineToken>>,
    truncation: Option<&TruncationParams>,
    num_added_special_tokens: usize,
) -> Result<(Vec<PipelineToken>, Option<Vec<PipelineToken>>)> {
    let truncated = truncate_pair_with_windows(s1, maybe_s2, truncation, num_added_special_tokens)?;
    Ok((truncated.first, truncated.second))
}

/// Truncate a pair like [`truncate_pair`], and return the windows each cut sequence tiles into.
///
/// A window is `keep` tokens long (the tokens the cut sequence keeps, `max_length` minus the
/// special tokens the post-processor adds, possibly split between the sides by `LongestFirst`)
/// and starts `keep - stride` tokens after the previous one, so consecutive windows share `stride`
/// tokens. The first window is the sequence itself; the rest come back here. Shapes and order are
/// released `Encoding::truncate`'s (`tokenizer/encoding.rs`): `Right` windows tile from the start
/// of the sequence, `Left` ones from its end with the windows coming back right to left, and the
/// tiling stops at the first window that reaches the far end, which can be shorter than `keep`.
///
/// A sequence that fits is not windowed at all. A sequence with no room (`keep == 0`, a
/// `max_length` no larger than the special tokens) is not a tiling but a single window holding
/// the whole sequence, exactly as released `Encoding::truncate(0, ..)` does. A `stride` of
/// `keep` or more can never advance a window and is a [`TruncationError::StrideTooLarge`], which
/// released panics on instead.
pub fn truncate_pair_with_windows(
    s1: Vec<PipelineToken>,
    maybe_s2: Option<Vec<PipelineToken>>,
    truncation: Option<&TruncationParams>,
    num_added_special_tokens: usize,
) -> Result<TruncatedPair> {
    let uncut = |first, second| TruncatedPair {
        first,
        first_windows: Vec::new(),
        second,
        second_windows: Vec::new(),
    };

    let Some(truncation) = truncation else {
        // None config = no truncation
        return Ok(uncut(s1, maybe_s2));
    };
    let seq_len = s1.len() + maybe_s2.as_ref().map_or(0, Vec::len);
    let truncate_to_length = truncation
        .max_length
        .saturating_sub(num_added_special_tokens);

    if truncate_to_length == 0 {
        // XXX: maybe we should error out when instantiating the PipelineTokenizer to avoid this
        warn!(
            "Truncation max_length is too short to include the tokens: `max_length` is {}, the post-processor adds {num_added_special_tokens} special tokens. Returning empty sequences: the input comes back whole as one overflowing window per sequence",
            truncation.max_length
        );
        let (first, first_windows) =
            truncate_tokens_with_windows(s1, 0, truncation.stride, truncation.direction)?;
        let (second, second_windows) = match maybe_s2 {
            Some(s2) => {
                let (second, windows) =
                    truncate_tokens_with_windows(s2, 0, truncation.stride, truncation.direction)?;
                (Some(second), windows)
            }
            None => (None, Vec::new()),
        };
        return Ok(TruncatedPair {
            first,
            first_windows,
            second,
            second_windows,
        });
    }

    if seq_len <= truncate_to_length {
        // No need to truncate
        return Ok(uncut(s1, maybe_s2));
    }
    let num_removed = seq_len - truncate_to_length;

    match truncation.strategy {
        TruncationStrategy::LongestFirst => {
            if let Some(s2) = maybe_s2 {
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
                // Both keeps are known here, so both strides are validated before either side
                // materializes its windows: an error on one side must not cost the other side's
                // copies on the way out.
                check_stride(s1.len(), n1, truncation.stride)?;
                check_stride(s2.len(), n2, truncation.stride)?;
                let (first, first_windows) =
                    truncate_tokens_with_windows(s1, n1, truncation.stride, truncation.direction)?;
                let (second, second_windows) =
                    truncate_tokens_with_windows(s2, n2, truncation.stride, truncation.direction)?;
                Ok(TruncatedPair {
                    first,
                    first_windows,
                    second: Some(second),
                    second_windows,
                })
            } else {
                let keep = s1.len() - num_removed;
                let (first, first_windows) = truncate_tokens_with_windows(
                    s1,
                    keep,
                    truncation.stride,
                    truncation.direction,
                )?;
                Ok(TruncatedPair {
                    first,
                    first_windows,
                    second: None,
                    second_windows: Vec::new(),
                })
            }
        }
        TruncationStrategy::OnlyFirst | TruncationStrategy::OnlySecond => {
            let (sequence_to_truncate, other) =
                if truncation.strategy == TruncationStrategy::OnlyFirst {
                    (s1, maybe_s2)
                } else if maybe_s2.is_some() {
                    (maybe_s2.unwrap(), Some(s1))
                } else {
                    return Err(Box::new(TruncationError::SecondSequenceNotProvided));
                };
            let sequence_length = sequence_to_truncate.len();
            if sequence_length > num_removed {
                let (kept, windows) = truncate_tokens_with_windows(
                    sequence_to_truncate,
                    sequence_length - num_removed,
                    truncation.stride,
                    truncation.direction,
                )?;
                if truncation.strategy == TruncationStrategy::OnlyFirst {
                    Ok(TruncatedPair {
                        first: kept,
                        first_windows: windows,
                        second: other,
                        second_windows: Vec::new(),
                    })
                } else {
                    Ok(TruncatedPair {
                        first: other.unwrap(),
                        first_windows: Vec::new(),
                        second: Some(kept),
                        second_windows: windows,
                    })
                }
            } else {
                Err(Box::new(TruncationError::SequenceTooShort))
            }
        }
    }
}

/// The stride check released `Encoding::truncate` asserts on: a stride of `keep` or more can
/// never advance a window. A sequence that fits is not windowed at all and a `keep` of 0 has no
/// window to advance, so neither is an error however large the stride is.
fn check_stride(len: usize, keep: usize, stride: usize) -> Result<()> {
    if keep > 0 && keep < len && stride >= keep {
        return Err(Box::new(TruncationError::StrideTooLarge {
            stride,
            room: keep,
        }));
    }
    Ok(())
}

/// Truncate one sequence to `keep` tokens and return the windows it tiles into. The first window
/// is the truncated sequence itself and is not returned here.
fn truncate_tokens_with_windows(
    mut tokens: Vec<PipelineToken>,
    keep: usize,
    stride: usize,
    direction: TruncationDirection,
) -> Result<(Vec<PipelineToken>, Vec<Vec<PipelineToken>>)> {
    let len = tokens.len();
    if keep >= len {
        return Ok((tokens, Vec::new()));
    }
    if keep == 0 {
        // Not a tiling but the shape released `Encoding::truncate(0, ..)` gives it: the sequence
        // comes back whole as its single window. No window to advance, so `stride` is moot here.
        return Ok((Vec::new(), vec![tokens]));
    }
    check_stride(len, keep, stride)?;
    let windows = window_ranges(len, keep, stride, direction)
        .into_iter()
        .skip(1)
        .map(|(start, stop)| tokens[start..stop].to_vec())
        .collect();
    truncate_tokens(&mut tokens, keep, direction);
    Ok((tokens, windows))
}

/// The `(start, stop)` of every window, first window first. Ported verbatim from released
/// `Encoding::truncate` (`tokenizer/encoding.rs`) so the two tilings cannot drift.
fn window_ranges(
    len: usize,
    keep: usize,
    stride: usize,
    direction: TruncationDirection,
) -> Vec<(usize, usize)> {
    let offset = keep - stride;
    let mut end = false;
    match direction {
        TruncationDirection::Right => (0..len)
            .step_by(offset)
            .filter_map(|start| {
                if !end {
                    let stop = std::cmp::min(start + keep, len);
                    end = stop == len;
                    Some((start, stop))
                } else {
                    None
                }
            })
            .collect(),
        TruncationDirection::Left => (0..len)
            .rev()
            .step_by(offset)
            .filter_map(|stop| {
                let stop = stop + 1;
                let start = stop.saturating_sub(keep);
                if start < stop && !end {
                    end = start == 0;
                    Some((start, stop))
                } else {
                    None
                }
            })
            .collect(),
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
        let (t1, t2) = truncate_pair(s1, Some(s2), truncation.as_ref(), 0).unwrap();
        assert_eq!(t1.len(), n1);
        assert_eq!(t2.expect("the pair is kept").len(), n2);
    }

    #[test]
    fn test_longest_first_pair() {
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
    fn test_no_truncation() {
        truncate_and_assert(long(), long(), &None, 8, 8);
    }

    #[test]
    fn test_longest_first_single() {
        let (t1, t2) = truncate_pair(
            long(),
            None,
            params(3, TruncationStrategy::LongestFirst).as_ref(),
            0,
        )
        .unwrap();

        assert_eq!(t1, make_tokens(7..10));
        assert!(t2.is_none());
    }

    // The specials the post-processor will add are not in either sequence yet, so the caller passes
    // their count and truncation has to make room for them.
    #[test]
    fn test_specials() {
        let params = params(8, TruncationStrategy::LongestFirst);

        let (untouched, _) = truncate_pair(long(), None, params.as_ref(), 0).unwrap();
        assert_eq!(untouched, long());

        let (shortened, _) = truncate_pair(long(), None, params.as_ref(), 2).unwrap();
        assert_eq!(shortened, make_tokens(7..13));
    }

    #[test]
    fn test_only_first() {
        let (t1, t2) = truncate_pair(
            long(),
            Some(short()),
            params(7, TruncationStrategy::OnlyFirst).as_ref(),
            0,
        )
        .unwrap();

        assert_eq!(t1, make_tokens(7..12));
        assert_eq!(t2, Some(short()));
    }

    #[test]
    fn test_only_second() {
        let (t1, t2) = truncate_pair(
            short(),
            Some(long()),
            params(7, TruncationStrategy::OnlySecond).as_ref(),
            0,
        )
        .unwrap();

        assert_eq!(t1, short());
        assert_eq!(t2, Some(make_tokens(7..12)));
    }

    // `OnlySecond` names the sequence to cut, so there is nothing to cut without a pair. Silently
    // falling back to the first sequence would truncate what the caller asked us to keep.
    #[test]
    fn test_only_second_no_pair() {
        let err = truncate_pair(
            long(),
            None,
            params(7, TruncationStrategy::OnlySecond).as_ref(),
            0,
        )
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
    fn test_only_first_too_short() {
        let err = truncate_pair(
            short(),
            Some(long()),
            params(1, TruncationStrategy::OnlyFirst).as_ref(),
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
    fn test_max_length_zero() {
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
    fn test_truncate_right() {
        assert_eq!(
            truncated(long(), 3, TruncationDirection::Right),
            make_tokens(7..10)
        );
    }

    #[test]
    fn test_truncate_left() {
        assert_eq!(
            truncated(long(), 3, TruncationDirection::Left),
            make_tokens(12..15)
        );
    }

    #[test]
    fn test_keep_len() {
        for direction in BOTH_DIRECTIONS {
            assert_eq!(truncated(long(), 8, direction), long());
        }
    }

    // The pair strategy can ask to keep more tokens than a sequence holds, which is why the left
    // branch saturates: `num_special_tokens` 10 against a 2 and 5 token pair with `max_length` 9
    // computes 7 tokens to keep out of the 5 the second sequence has.
    #[test]
    fn test_keep_more_than_len() {
        for direction in BOTH_DIRECTIONS {
            assert_eq!(truncated(medium(), 9, direction), medium());
            assert_eq!(truncated(empty(), 9, direction), empty());
        }
    }

    #[test]
    fn test_keep_zero() {
        for direction in BOTH_DIRECTIONS {
            assert!(truncated(long(), 0, direction).is_empty());
        }
    }

    // Draining the head keeps the buffer the tokens are already in. Collecting the tail into a new
    // `Vec` would pay an allocation and a copy instead, which is the whole reason the left branch
    // is written as a drain.
    #[test]
    fn test_left_reuses_allocation() {
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
        let (t1, t2) = truncate_pair(s1, s2, Some(&truncation), num_special_tokens).unwrap();

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
    fn test_direction_left() {
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
    fn test_specials_pair() {
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

    // The windows are the released `Encoding::truncate`'s, of which `tokenizer/encoding.rs` keeps
    // the last independent copy in the tree: the released crate itself cannot serve as a full
    // encode oracle for windows because it leaves their type ids at 0 (issue #1908) and nests
    // duplicates of them inside each other. Every shape below is compared against it.
    use crate::tokenizer::Encoding as LegacyEncoding;

    fn legacy_truncates(
        input: &[u32],
        keep: usize,
        stride: usize,
        direction: TruncationDirection,
    ) -> (Vec<u32>, Vec<Vec<u32>>) {
        let len = input.len();
        let mut reference = LegacyEncoding::new(
            input.to_vec(),
            vec![0; len],
            vec![String::new(); len],
            vec![None; len],
            vec![(0, 0); len],
            vec![0; len],
            vec![1; len],
            Vec::new(),
            Default::default(),
        );
        reference.truncate(keep, stride, direction);
        let windows = reference
            .get_overflowing()
            .iter()
            .map(|window| window.get_ids().to_vec())
            .collect();
        (reference.get_ids().to_vec(), windows)
    }

    fn windows_of(
        input: &[u32],
        keep: usize,
        stride: usize,
        direction: TruncationDirection,
    ) -> (Vec<u32>, Vec<Vec<u32>>) {
        let (main, windows) = truncate_tokens_with_windows(
            make_tokens(input.iter().copied()),
            keep,
            stride,
            direction,
        )
        .unwrap();
        let ids = |tokens: &Vec<PipelineToken>| tokens.iter().map(|token| token.id()).collect();
        (ids(&main), windows.iter().map(ids).collect())
    }

    #[test]
    fn windows_match_the_legacy_truncation() {
        for direction in BOTH_DIRECTIONS {
            for (len, keep, stride) in [
                (5usize, 4usize, 0usize), // tiling with a last window shorter than the rest
                (5, 4, 2),                // the `truncate_overflow_with_stride` case
                (6, 2, 0),                // exact tiling
                (8, 3, 1),                // overlap with a short last window
                (3, 3, 0),                // fits: no windows
                (4, 9, 0),                // roomier than the input: no windows
                (5, 0, 0),                // no room: the whole sequence is one window
            ] {
                let input: Vec<u32> = (0..len as u32).map(|i| 100 + i).collect();
                assert_eq!(
                    windows_of(&input, keep, stride, direction),
                    legacy_truncates(&input, keep, stride, direction),
                    "len={len} keep={keep} stride={stride} direction={direction:?}"
                );
            }
        }
    }

    // Released `Encoding::truncate` asserts `stride < max_len` and panics; the error has to name
    // the room each side actually windows at (`LongestFirst` splits it between the sides).
    #[test]
    fn a_stride_at_least_the_room_is_an_error_when_windowing() {
        for (keep, stride) in [(3usize, 3usize), (3, 5), (1, 1)] {
            let err = truncate_tokens_with_windows(
                make_tokens(1..=8),
                keep,
                stride,
                TruncationDirection::Right,
            )
            .err()
            .unwrap();
            match err.downcast_ref::<TruncationError>() {
                Some(TruncationError::StrideTooLarge {
                    stride: err_stride,
                    room,
                }) => assert_eq!((*err_stride, *room), (stride, keep)),
                other => panic!("expected StrideTooLarge, got {other:?}"),
            }
        }
    }

    // The legacy assert sits after its `max_len >= len` early return, so a sequence that fits is
    // not windowed and never trips it however large the stride is.
    #[test]
    fn a_stride_at_least_the_room_is_fine_while_the_sequence_fits() {
        let (main, windows) =
            truncate_tokens_with_windows(make_tokens(1..=2), 3, 5, TruncationDirection::Right)
                .unwrap();

        assert_eq!(main, make_tokens(1..=2));
        assert!(windows.is_empty());
    }

    #[test]
    fn no_room_takes_the_whole_sequence_as_one_window() {
        let (main, windows) =
            truncate_tokens_with_windows(make_tokens(1..=5), 0, 7, TruncationDirection::Right)
                .unwrap();

        assert!(main.is_empty());
        assert_eq!(windows, vec![make_tokens(1..=5)]);
    }

    // Each side windows on its own: `max_length` 7 minus 3 pair specials leaves 4, which
    // `LongestFirst` splits 2 and 2.
    #[test]
    fn each_side_windows_on_its_own() {
        let truncated = truncate_pair_with_windows(
            make_tokens(1..=4),
            Some(make_tokens(11..=14)),
            Some(TruncationParams {
                max_length: 7,
                strategy: TruncationStrategy::LongestFirst,
                ..TruncationParams::default()
            })
            .as_ref(),
            3,
        )
        .unwrap();

        assert_eq!(truncated.first, make_tokens(1..=2));
        assert_eq!(truncated.first_windows, vec![make_tokens(3..=4)]);
        assert_eq!(truncated.second, Some(make_tokens(11..=12)));
        assert_eq!(truncated.second_windows, vec![make_tokens(13..=14)]);
    }

    // A `LongestFirst` pair can only reach its second side's stride check after the first side's
    // keep is known -- but both keeps are known before either side copies a single window. This
    // pins the preflight: the error must come back before the other side's windows are
    // materialized, which would be copies the error throws away again.
    #[test]
    fn a_pair_error_does_not_materialize_the_other_sides_windows() {
        // The 41-token room splits 21 and 20 between the sides, so a stride of 20 can window the
        // first side but can never advance the second one's.
        let params = TruncationParams {
            max_length: 44,
            stride: 20,
            ..TruncationParams::default()
        };

        let before = allocation_probe::allocated_bytes();
        let err = truncate_pair_with_windows(
            make_tokens(0..5000),
            Some(make_tokens(0..1500)),
            Some(&params),
            3,
        )
        .err()
        .unwrap();
        let allocated = allocation_probe::allocated_bytes() - before;

        assert!(matches!(
            err.downcast_ref::<TruncationError>(),
            Some(TruncationError::StrideTooLarge {
                stride: 20,
                room: 20
            })
        ));
        // Materializing the first side first would tile 5000 tokens into ~4980 windows of 21,
        // about 420 KiB of copies. The preflight errors before any of them, so only the
        // bookkeeping around the error is ever allocated. Bounded on purpose: no OOM probe.
        assert!(
            allocated < 100_000,
            "allocated {allocated} bytes before the second side's error"
        );
    }

    // `OnlyFirst` and `OnlySecond` name one sequence to cut, so only that one gets windows and
    // the other comes through untouched.
    #[test]
    fn only_first_and_only_second_window_one_side_each() {
        let params = |strategy| {
            Some(TruncationParams {
                max_length: 8,
                strategy,
                ..TruncationParams::default()
            })
        };

        let first = truncate_pair_with_windows(
            make_tokens(1..=6),
            Some(make_tokens(11..=13)),
            params(TruncationStrategy::OnlyFirst).as_ref(),
            1,
        )
        .unwrap();
        // room = 7, 2 to remove from the first sequence: it keeps 4 and windows 2.
        assert_eq!(first.first, make_tokens(1..=4));
        assert_eq!(first.first_windows, vec![make_tokens(5..=6)]);
        assert_eq!(first.second, Some(make_tokens(11..=13)));
        assert!(first.second_windows.is_empty());

        let second = truncate_pair_with_windows(
            make_tokens(1..=6),
            Some(make_tokens(11..=13)),
            params(TruncationStrategy::OnlySecond).as_ref(),
            1,
        )
        .unwrap();
        // The same 2 to remove from the three-token pair: it keeps 1 and windows 2.
        assert_eq!(second.first, make_tokens(1..=6));
        assert!(second.first_windows.is_empty());
        assert_eq!(second.second, Some(make_tokens(11..=11)));
        assert_eq!(
            second.second_windows,
            vec![make_tokens(12..=12), make_tokens(13..=13)]
        );
    }
}

/// Counts what one test thread allocates, so a test can pin that an erroring call does not
/// materialize windows on the way to its error. Thread-local because the test harness runs
/// tests in parallel: each sees only its own allocations. Test binary only, never in the lib.
#[cfg(test)]
mod allocation_probe {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        static ALLOCATED_BYTES: Cell<usize> = const { Cell::new(0) };
    }

    pub struct CountingAlloc;

    unsafe impl GlobalAlloc for CountingAlloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            ALLOCATED_BYTES.with(|bytes| bytes.set(bytes.get() + layout.size()));
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            ALLOCATED_BYTES.with(|bytes| bytes.set(bytes.get() + new_size));
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }

    #[global_allocator]
    static ALLOCATOR: CountingAlloc = CountingAlloc;

    pub fn allocated_bytes() -> usize {
        ALLOCATED_BYTES.with(Cell::get)
    }
}
