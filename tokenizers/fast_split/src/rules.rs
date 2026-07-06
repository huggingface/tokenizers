use crate::matchers::Matcher;

/// Match a run of `M` between MIN and MAX chars. `MAX == u8::MAX` means unbounded (`+`/`*`).
/// Steps one char at a time and stops at MAX — never scans past what it returns, so it's
/// O(MAX) per call (O(1) for bounded), not O(run length). `MAX == u8::MAX` const-folds away
/// the cap for the unbounded case.
pub fn run_matcher<M: Matcher, const MIN: u8, const MAX: u8>(
    b: &[u8],
    from: usize,
) -> Option<usize> {
    let mut pos = from;
    if MAX == u8::MAX {
        // unbounded (+/*): step to the run end, no per-char counting.
        while let Some(end) = M::step(b, pos) {
            pos = end;
        }
        // MIN is 0 (`*`) or 1 (`+`); either MIN==0 or we need at least one char.
        if MIN == 0 || pos > from {
            Some(pos)
        } else {
            None
        }
    } else {
        // bounded ({m,n}): stop after MAX chars.
        let mut count = 0usize;
        while count < MAX as usize {
            match M::step(b, pos) {
                Some(end) => {
                    pos = end;
                    count += 1;
                }
                None => break,
            }
        }
        if count >= MIN as usize {
            Some(pos)
        } else {
            None
        }
    }
}

pub fn word_rule<L: Matcher>(b: &[u8], from: usize) -> Option<usize> {
    // a word is `\p{L}+`: one or more, unbounded.
    run_matcher::<L, 1, { u8::MAX }>(b, from)
}

#[cfg(all(test, feature = "unicode"))]
mod tests {
    use super::*;

    use crate::matchers::{FastLetter, Letter};
    #[test]
    fn test_run_matcher() {
        let string = "This is 123 anditshouldbenotedthat 12".as_bytes();
        //                      3      9                        34
        assert_eq!(run_matcher::<Letter, 1, 3>(string, 0), Some(3));
        assert_eq!(run_matcher::<FastLetter, 1, 3>(string, 0), Some(3));
        assert_eq!(run_matcher::<Letter, 0, 255>(string, 0), Some(4));
        assert_eq!(run_matcher::<Letter, 0, 255>(string, 9), Some(9)); // because 0 repetition of a letter :)
        assert_eq!(run_matcher::<Letter, 0, 255>(string, 15), Some(34));
        // bounded + from > 0  → catches the relative-offset bug (from+off, not off)
        // at 15 = "itshould…"; cap 3 letters "its" → end 18
        assert_eq!(run_matcher::<Letter, 1, 3>(string, 15), Some(18));
        // bounded run SHORTER than MAX
        // at 5 = "is" (2 letters ≤ 3) → take 2 → end 7
        assert_eq!(run_matcher::<Letter, 1, 3>(string, 5), Some(7));
        // MIN=1 on a non-letter → catches the empty-MIN bug
        // at 8 = '1' (digit) → no letter, MIN=1 → None
        assert_eq!(run_matcher::<Letter, 1, 255>(string, 8), None);
    }
} // non-ASCII we use bitmap_generated entires.
