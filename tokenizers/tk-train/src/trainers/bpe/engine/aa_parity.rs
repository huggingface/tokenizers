//! Exact left-to-right AA selection from ordered chunks of valid pair starts.
//! AA means `(A, A)` for one token identity. `token_length` measures A's span
//! in fixed corpus slots. Edge-run parity here is independent of the optional
//! parity-aware BPE trainer; the coordinator processes summaries sequentially.
//!
//! Each chunk locates its trailing run by logarithmic boundary checks. The coordinator
//! processes one summary per chunk; workers then select their own starts using
//! the returned incoming parity. Empty chunks preserve the previous run state.
//! No corpus-sized marking array or coordinator occurrence scan is needed.
//!
//! Preconditions: starts are globally strictly increasing, are valid AA edges
//! from one stable corpus snapshot, their gaps are >= token_length,
//! and the token's length is positive. Pieces
//! separated by sentinels must have a physical gap greater than token_length.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RunSummary {
    pub first: u64,
    pub last: u64,
    /// Parity of the number of AA edges in the final run of this chunk.
    pub trailing_odd: bool,
    pub all_one_run: bool,
}

#[inline]
fn consecutive(left: u64, right: u64, token_length: u64) -> bool {
    left.checked_add(token_length) == Some(right)
}

/// Inspect a fixed chunk without a full summary scan. Valid, sorted AA starts
/// have gaps >= token_length. Consequently the suffix is one run exactly when
/// `last - starts[i] == (len - 1 - i) * token_length`. This predicate is monotone:
/// binary search finds the trailing run's first index in O(log len).
pub fn summarize_by(
    len: usize,
    at: impl Fn(usize) -> u64,
    token_length: u64,
) -> Option<RunSummary> {
    assert!(token_length > 0);
    if len == 0 {
        return None;
    }
    let first = at(0);
    let last = at(len - 1);
    let suffix_is_run =
        |i: usize| last.checked_sub(at(i)) == ((len - 1 - i) as u64).checked_mul(token_length);
    let all_one_run = suffix_is_run(0);
    let mut begin = 0;
    let mut end = len - 1;
    if !all_one_run {
        // Small trailing runs occupy nearby cache lines. Search those directly
        // before probing distant midpoint positions of a long trailing run.
        for _ in 0..8 {
            if end == 0 {
                break;
            }
            if !consecutive(at(end - 1), at(end), token_length) {
                return Some(RunSummary {
                    first,
                    last,
                    trailing_odd: (len - end) % 2 == 1,
                    all_one_run: false,
                });
            }
            end -= 1;
        }
        while begin < end {
            let mid = begin + (end - begin) / 2;
            if suffix_is_run(mid) {
                end = mid;
            } else {
                begin = mid + 1;
            }
        }
    }
    Some(RunSummary {
        first,
        last,
        trailing_odd: (len - begin) % 2 == 1,
        all_one_run,
    })
}

#[cfg(test)]
fn summarize(starts: &[u64], token_length: u64) -> Option<RunSummary> {
    summarize_by(starts.len(), |i| starts[i], token_length)
}

/// Returns whether each chunk must skip its first valid AA edge.
///
/// This is O(chunks) coordinator work, independent of the occurrence count.
/// Only the leading run uses incoming parity. Once a gap is encountered, the
/// worker restarts at the leftmost edge of that new run.
pub fn incoming_parities(summaries: &[Option<RunSummary>], token_length: u64) -> Vec<bool> {
    assert!(token_length > 0);
    let mut last = None;
    let mut trailing_odd = false;
    summaries
        .iter()
        .map(|summary| {
            let Some(summary) = summary else {
                return false;
            };
            debug_assert!(last.is_none_or(|previous| previous < summary.first));
            let incoming = last.is_some_and(|previous| {
                consecutive(previous, summary.first, token_length) && trailing_odd
            });
            trailing_odd = if summary.all_one_run {
                incoming ^ summary.trailing_odd
            } else {
                summary.trailing_odd
            };
            last = Some(summary.last);
            incoming
        })
        .collect()
}

/// Run on each chunk in parallel after its incoming parity is known.
/// The callback can directly build a worker's plan; no extra selected Vec is
/// required by this module. All callbacks must finish planning before writes.
pub fn for_each_selected(
    starts: impl IntoIterator<Item = u64>,
    token_length: u64,
    incoming_odd: bool,
    mut emit: impl FnMut(u64),
) {
    assert!(token_length > 0);
    let mut previous = None;
    let mut skip = incoming_odd;
    for pos in starts {
        if let Some(last) = previous {
            debug_assert!(pos > last);
            if !consecutive(last, pos, token_length) {
                skip = false;
            }
        }
        if !skip {
            emit(pos);
        }
        skip = !skip;
        previous = Some(pos);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(starts: &[u64], length: u64, cuts: &[usize]) {
        let chunks: Vec<_> = cuts.windows(2).map(|w| &starts[w[0]..w[1]]).collect();
        let summaries: Vec<_> = chunks.iter().map(|s| summarize(s, length)).collect();
        let incoming = incoming_parities(&summaries, length);
        let mut actual = Vec::new();
        for (chunk, odd) in chunks.iter().zip(incoming) {
            for_each_selected(chunk.iter().copied(), length, odd, |pos| actual.push(pos));
        }
        // Independent sequential greedy non-overlap, using the span of two A
        // tokens rather than the parity recurrence under test.
        let mut expected = Vec::new();
        let mut after = 0_u128;
        for &pos in starts {
            if pos as u128 >= after {
                expected.push(pos);
                after = pos as u128 + 2 * length as u128;
            }
        }
        assert_eq!(actual, expected, "length={length}, cuts={cuts:?}");
    }

    #[test]
    fn preserves_runs_over_empty_chunks_and_restarts_after_gaps() {
        let starts = [10, 11, 12, 20, 21, 22, 23, 24, 30];
        check(&starts, 1, &[0, 1, 1, 2, 5, 5, 8, 9]);
        check(&[], 1, &[0, 0, 0]);
        check(&[1], 512, &[0, 0, 1, 1]);
    }

    #[test]
    fn long_mixed_runs_match_linear_summary() {
        let mut starts = Vec::new();
        let mut position = 1_u64;
        for i in 0..17000 {
            position += if i % 4093 == 0 { 17 } else { 3 };
            starts.push(position);
        }
        for end in [2, 8, 31, 4096, 8203, 16999] {
            let slice = &starts[..end];
            let mut trailing = 1;
            for w in slice.windows(2).rev() {
                if w[1] - w[0] != 3 {
                    break;
                }
                trailing += 1;
            }
            let expected = RunSummary {
                first: slice[0],
                last: slice[end - 1],
                trailing_odd: trailing % 2 == 1,
                all_one_run: trailing == end,
            };
            assert_eq!(summarize_by(end, |i| slice[i], 3), Some(expected));
        }
    }

    #[test]
    fn long_tokens_and_large_positions_do_not_wrap() {
        check(&[1, 513, 1025, 1537, 3000, 3512], 512, &[0, 1, 3, 3, 5, 6]);
        check(
            &[u64::MAX - 20, u64::MAX - 15, u64::MAX - 5],
            5,
            &[0, 1, 1, 2, 3],
        );
        assert!(!consecutive(u64::MAX - 1, 3, 5));
    }

    #[test]
    fn all_partitions_of_small_valid_edge_patterns_match_greedy() {
        // A clear bit continues an AA run; a set bit leaves a gap. Every possible
        // chunk boundary, including duplicated boundaries for empty chunks,
        // is checked against the sequential greedy oracle.
        {
            let length = 1;
            for gaps in 0_usize..128 {
                let mut starts = vec![1_u64];
                for bit in 0..7 {
                    let gap = if gaps & (1 << bit) == 0 {
                        length
                    } else {
                        3 * length
                    };
                    starts.push(starts.last().unwrap() + gap);
                }
                for partitions in 0_usize..128 {
                    let mut cuts = vec![0];
                    for i in 1..8 {
                        if partitions & (1 << (i - 1)) != 0 {
                            cuts.extend([i, i]);
                        }
                    }
                    cuts.push(8);
                    check(&starts, length, &cuts);
                }
            }
        }
    }
}
