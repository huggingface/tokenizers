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
    /// A window that advances by `max_length - stride` tokens has to advance by at least one.
    #[error(
        "Truncation error: `stride` ({stride}) must be strictly less than the room left for the sequence ({room}: `max_length` minus the special tokens the post-processor adds)"
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

/// One window over a pair: the two sides of a window, in the shape the post-processor takes.
pub type Window = (Vec<PipelineToken>, Option<Vec<PipelineToken>>);

/// What truncation left: the window that becomes the encoding, and the windows it cut off.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Truncated {
    pub first: Vec<PipelineToken>,
    pub second: Option<Vec<PipelineToken>>,
    /// The removed tokens, as complete windows in reading order (see [`truncate_windows`]).
    /// Empty when nothing was removed. A pair yields one window per combination of its sides'
    /// windows, in the order released `tokenizers` 0.23 emitted them: every window of the first
    /// side against the kept second side and then against each window of the second side, then
    /// the kept first side against each window of the second side.
    pub overflowing: Vec<Window>,
}

impl Truncated {
    fn untouched(first: Vec<PipelineToken>, second: Option<Vec<PipelineToken>>) -> Self {
        Self {
            first,
            second,
            overflowing: Vec::new(),
        }
    }
}

pub fn truncate_pair(
    mut s1: Vec<PipelineToken>,
    maybe_s2: Option<Vec<PipelineToken>>,
    truncation: Option<&TruncationParams>,
    num_added_special_tokens: usize,
) -> Result<Truncated> {
    let seq_len = s1.len() + maybe_s2.as_ref().map_or(0, Vec::len);

    // None config = no truncation
    let Some(truncation) = truncation else {
        return Ok(Truncated::untouched(s1, maybe_s2));
    };
    let truncate_to_length = truncation
        .max_length
        .saturating_sub(num_added_special_tokens);

    if truncate_to_length == 0 {
        // XXX: maybe we should error out when instantiating the PipelineTokenizer to avoid this
        warn!(
            "Truncation max_length is too short to include the tokens: `max_length` is {}, the post-processor adds {num_added_special_tokens} special tokens. Returning an empty sequence",
            truncation.max_length
        );
        // Nothing fits, so everything overflows: one window holding the whole input, the way
        // released `tokenizers` moved a `max_len == 0` encoding into `overflowing` whole.
        let second = maybe_s2.as_ref().map(|_| Vec::new());
        let overflowing = if seq_len == 0 {
            Vec::new()
        } else {
            vec![(s1, maybe_s2)]
        };
        return Ok(Truncated {
            first: Vec::new(),
            second,
            overflowing,
        });
    }
    if seq_len <= truncate_to_length {
        // No need to truncate
        return Ok(Truncated::untouched(s1, maybe_s2));
    }
    let num_removed = seq_len - truncate_to_length;
    let (stride, direction) = (truncation.stride, truncation.direction);

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
                let windows1 = truncate_windows(&mut s1, n1, stride, direction)?;
                let windows2 = truncate_windows(&mut s2, n2, stride, direction)?;
                let mut overflowing =
                    Vec::with_capacity(windows1.len() * (1 + windows2.len()) + windows2.len());
                for w1 in &windows1 {
                    overflowing.push((w1.clone(), Some(s2.clone())));
                    for w2 in &windows2 {
                        overflowing.push((w1.clone(), Some(w2.clone())));
                    }
                }
                for w2 in windows2 {
                    overflowing.push((s1.clone(), Some(w2)));
                }
                Ok(Truncated {
                    first: s1,
                    second: Some(s2),
                    overflowing,
                })
            } else {
                let len = s1.len();
                let windows = truncate_windows(&mut s1, len - num_removed, stride, direction)?;
                Ok(Truncated {
                    first: s1,
                    second: None,
                    overflowing: windows.into_iter().map(|w| (w, None)).collect(),
                })
            }
        }
        TruncationStrategy::OnlyFirst | TruncationStrategy::OnlySecond => {
            let (mut sequence_to_truncate, other) =
                if truncation.strategy == TruncationStrategy::OnlyFirst {
                    (s1, maybe_s2)
                } else if let Some(s2) = maybe_s2 {
                    (s2, Some(s1))
                } else {
                    return Err(Box::new(TruncationError::SecondSequenceNotProvided));
                };
            let sequence_length = sequence_to_truncate.len();
            if sequence_length <= num_removed {
                return Err(Box::new(TruncationError::SequenceTooShort));
            }
            let windows = truncate_windows(
                &mut sequence_to_truncate,
                sequence_length - num_removed,
                stride,
                direction,
            )?;
            if truncation.strategy == TruncationStrategy::OnlyFirst {
                let overflowing = windows.into_iter().map(|w| (w, other.clone())).collect();
                Ok(Truncated {
                    first: sequence_to_truncate,
                    second: other,
                    overflowing,
                })
            } else {
                // `other` is `Some`: the branch above only reaches here with a pair.
                let first = other.unwrap_or_default();
                let overflowing = windows
                    .into_iter()
                    .map(|w| (first.clone(), Some(w)))
                    .collect();
                Ok(Truncated {
                    first,
                    second: Some(sequence_to_truncate),
                    overflowing,
                })
            }
        }
    }
}

/// Cuts `tokens` down to `keep` tokens in place and returns the windows that cover what was
/// removed, in reading order.
///
/// Windows are `keep` tokens wide and each one starts `keep - stride` tokens after the previous,
/// so with a `stride` of 0 they tile the sequence and with a positive one each window repeats
/// the last `stride` tokens of the one before it. The first window is the one kept in `tokens`;
/// the last window is cut short at the end of the sequence rather than padded. `Left` walks the
/// same windows from the end of the sequence, so the kept window is its tail. Same windows as
/// released `tokenizers` 0.23 `Encoding::truncate`.
///
/// A `stride` that is not smaller than `keep` would never advance, which is an error rather
/// than the assertion it used to be, since both come from a config.
pub fn truncate_windows(
    tokens: &mut Vec<PipelineToken>,
    keep: usize,
    stride: usize,
    direction: TruncationDirection,
) -> Result<Vec<Vec<PipelineToken>>> {
    let len = tokens.len();
    if keep >= len {
        return Ok(Vec::new());
    }
    if keep == 0 {
        return Ok(vec![mem::take(tokens)]);
    }
    if stride >= keep {
        return Err(Box::new(TruncationError::StrideTooLarge {
            stride,
            room: keep,
        }));
    }
    let step = keep - stride;
    let mut windows = Vec::with_capacity((len - keep).div_ceil(step));
    match direction {
        TruncationDirection::Right => {
            let mut start = step;
            loop {
                let stop = cmp::min(start + keep, len);
                windows.push(tokens[start..stop].to_vec());
                if stop == len {
                    break;
                }
                start += step;
            }
            tokens.truncate(keep);
        }
        TruncationDirection::Left => {
            let mut stop = len - step;
            loop {
                let start = stop.saturating_sub(keep);
                windows.push(tokens[start..stop].to_vec());
                if start == 0 {
                    break;
                }
                stop -= step;
            }
            tokens.drain(..len - keep);
        }
    }
    Ok(windows)
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

    /// The kept pair, dropping the windows: for the tests that only check lengths.
    fn truncate_kept(
        s1: Vec<PipelineToken>,
        s2: Option<Vec<PipelineToken>>,
        truncation: Option<&TruncationParams>,
        num_added_special_tokens: usize,
    ) -> Result<(Vec<PipelineToken>, Option<Vec<PipelineToken>>)> {
        let Truncated { first, second, .. } =
            truncate_pair(s1, s2, truncation, num_added_special_tokens)?;
        Ok((first, second))
    }

    fn truncate_and_assert(
        s1: Vec<PipelineToken>,
        s2: Vec<PipelineToken>,
        truncation: &Option<TruncationParams>,
        n1: usize,
        n2: usize,
    ) {
        let (t1, t2) = truncate_kept(s1, Some(s2), truncation.as_ref(), 0).unwrap();
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
        let (t1, t2) = truncate_kept(
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

        let (untouched, _) = truncate_kept(long(), None, params.as_ref(), 0).unwrap();
        assert_eq!(untouched, long());

        let (shortened, _) = truncate_kept(long(), None, params.as_ref(), 2).unwrap();
        assert_eq!(shortened, make_tokens(7..13));
    }

    #[test]
    fn test_only_first() {
        let (t1, t2) = truncate_kept(
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
        let (t1, t2) = truncate_kept(
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
        let err = truncate_kept(
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
        let err = truncate_kept(
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
        truncate_windows(&mut tokens, keep, 0, direction).unwrap();
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

        truncate_windows(&mut tokens, 3, 0, TruncationDirection::Left).unwrap();

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
        let (t1, t2) = truncate_kept(s1, s2, Some(&truncation), num_special_tokens).unwrap();

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

    fn windows_of(
        tokens: Vec<PipelineToken>,
        keep: usize,
        stride: usize,
        direction: TruncationDirection,
    ) -> (Vec<u32>, Vec<Vec<u32>>) {
        let mut tokens = tokens;
        let windows = truncate_windows(&mut tokens, keep, stride, direction).unwrap();
        let ids = |t: &[PipelineToken]| t.iter().map(|t| t.id()).collect::<Vec<_>>();
        (ids(&tokens), windows.iter().map(|w| ids(w)).collect())
    }

    // The windows released tokenizers 0.23.1 `Encoding::truncate` produces: with no stride they
    // tile the sequence, the last one cut short; with a stride each repeats the tail of the one
    // before. `Left` walks the same windows from the end, so the kept one is the tail.
    #[test]
    fn test_windows_tile_and_overlap() {
        assert_eq!(
            windows_of(long(), 3, 0, TruncationDirection::Right),
            (vec![7, 8, 9], vec![vec![10, 11, 12], vec![13, 14]])
        );
        assert_eq!(
            windows_of(long(), 3, 1, TruncationDirection::Right),
            (
                vec![7, 8, 9],
                vec![vec![9, 10, 11], vec![11, 12, 13], vec![13, 14]]
            )
        );
        assert_eq!(
            windows_of(long(), 3, 0, TruncationDirection::Left),
            (vec![12, 13, 14], vec![vec![9, 10, 11], vec![7, 8]])
        );
        assert_eq!(
            windows_of(long(), 3, 1, TruncationDirection::Left),
            (
                vec![12, 13, 14],
                vec![vec![10, 11, 12], vec![8, 9, 10], vec![7, 8]]
            )
        );
        // Exactly divisible: no short last window, and no empty one either.
        assert_eq!(
            windows_of(medium(), 2, 0, TruncationDirection::Right),
            (vec![3, 4], vec![vec![5, 6]])
        );
        // Nothing to cut: no windows.
        assert_eq!(
            windows_of(short(), 5, 0, TruncationDirection::Right),
            (vec![1, 2], vec![])
        );
    }

    // A stride that does not advance the window is a config error, not a hang or an assertion.
    #[test]
    fn test_stride_must_advance() {
        let mut tokens = long();
        let err = truncate_windows(&mut tokens, 3, 3, TruncationDirection::Right)
            .err()
            .unwrap();
        assert!(matches!(
            err.downcast_ref::<TruncationError>(),
            Some(TruncationError::StrideTooLarge { stride: 3, room: 3 })
        ));
        // But only when there is something to cut: a short sequence never reaches the check.
        assert!(truncate_windows(&mut short(), 3, 3, TruncationDirection::Right).is_ok());
    }

    fn window_ids(windows: &[Window]) -> Vec<(Vec<u32>, Option<Vec<u32>>)> {
        let ids = |t: &[PipelineToken]| t.iter().map(|t| t.id()).collect::<Vec<_>>();
        windows
            .iter()
            .map(|(a, b)| (ids(a), b.as_deref().map(ids)))
            .collect()
    }

    // A pair overflows as the combinations released tokenizers 0.23.1 `merge_with` built, in its
    // order: each first-side window with the kept second side, then with every second-side
    // window, and finally the kept first side with every second-side window.
    #[test]
    fn test_pair_windows_combine() {
        let t = truncate_pair(
            long(),
            Some(long()),
            params(4, TruncationStrategy::LongestFirst).as_ref(),
            0,
        )
        .unwrap();
        assert_eq!(t.first, make_tokens(7..9));
        assert_eq!(t.second, Some(make_tokens(7..9)));
        assert_eq!(
            window_ids(&t.overflowing),
            vec![
                (vec![9, 10], Some(vec![7, 8])),
                (vec![9, 10], Some(vec![9, 10])),
                (vec![9, 10], Some(vec![11, 12])),
                (vec![9, 10], Some(vec![13, 14])),
                (vec![11, 12], Some(vec![7, 8])),
                (vec![11, 12], Some(vec![9, 10])),
                (vec![11, 12], Some(vec![11, 12])),
                (vec![11, 12], Some(vec![13, 14])),
                (vec![13, 14], Some(vec![7, 8])),
                (vec![13, 14], Some(vec![9, 10])),
                (vec![13, 14], Some(vec![11, 12])),
                (vec![13, 14], Some(vec![13, 14])),
                (vec![7, 8], Some(vec![9, 10])),
                (vec![7, 8], Some(vec![11, 12])),
                (vec![7, 8], Some(vec![13, 14])),
            ]
        );

        // `OnlySecond` keeps the first side whole in every window.
        let t = truncate_pair(
            short(),
            Some(long()),
            params(7, TruncationStrategy::OnlySecond).as_ref(),
            0,
        )
        .unwrap();
        assert_eq!(t.first, short());
        assert_eq!(t.second, Some(make_tokens(7..12)));
        assert_eq!(
            window_ids(&t.overflowing),
            vec![(vec![1, 2], Some(vec![12, 13, 14]))]
        );

        // A single sequence overflows plain windows.
        let t = truncate_pair(
            long(),
            None,
            params(3, TruncationStrategy::LongestFirst).as_ref(),
            0,
        )
        .unwrap();
        assert_eq!(
            window_ids(&t.overflowing),
            vec![(vec![10, 11, 12], None), (vec![13, 14], None)]
        );
    }

    // When the specials alone fill `max_length`, everything overflows as one window and the pair
    // still comes back as a pair.
    #[test]
    fn test_no_room_overflows_whole() {
        let t = truncate_pair(
            short(),
            Some(medium()),
            params(2, TruncationStrategy::LongestFirst).as_ref(),
            2,
        )
        .unwrap();
        assert!(t.first.is_empty());
        assert_eq!(t.second, Some(Vec::new()));
        assert_eq!(
            window_ids(&t.overflowing),
            vec![(vec![1, 2], Some(vec![3, 4, 5, 6]))]
        );
        let t = truncate_pair(
            empty(),
            None,
            params(2, TruncationStrategy::LongestFirst).as_ref(),
            2,
        )
        .unwrap();
        assert!(t.overflowing.is_empty());
    }
}
