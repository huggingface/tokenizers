use crate::matchers::{Letter, Matcher};

pub fn run_matcher<M: Matcher, const MIN: u8, const MAX: u8>(
    b: &[u8],
    from: usize,
) -> Option<usize> {
    // If MAX is bounded we do scallar checks.
    let longest = M::run_end(b, from);
    if longest == from {
        // matcher stopped at the first char as it did not match
        return if MIN == 0 { Some(from) } else { None };
    }
    if MAX == u8::MAX {
        return Some(longest); // longest > form there's at least 1 match maybe more
    }
    // Now the bounds
    match unsafe { str::from_utf8_unchecked(&b[from..longest]) }
        .char_indices()
        .nth(MAX as usize)
    {
        Some((off, _)) => Some(from + off),
        _ => Some(longest),
    }
}

pub fn word_rule<L: Matcher>(b: &[u8], from: usize) -> Option<usize> {
    // the end of a word depends of course on the matcher.
    let i = L::run_end(b, from);
    if i == from { None } else { Some(i) }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::matchers::Letter;
    #[test]
    fn test_run_matcher() {
        let string = "This is 123 anditshouldbenotedthat 12".as_bytes();
        //                      3      9                        34
        assert_eq!(run_matcher::<Letter, 1, 3>(string, 0), Some(3));
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
