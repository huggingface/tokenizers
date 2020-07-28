use crate::{Offsets, Result};
use regex::Regex;

/// Pattern used to split a NormalizedString
pub trait Pattern {
    /// Slice the given string in a list of pattern match positions, with
    /// a boolean indicating whether this is a match or not.
    ///
    /// This method *must* cover the whole string in its outputs, with
    /// contiguous ordered slices.
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>>;
}

impl Pattern for char {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        let is_char = |c: char| -> bool { c == *self };
        is_char.find_matches(inside)
    }
}

impl Pattern for &str {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        let re = Regex::new(&regex::escape(self))?;
        (&re).find_matches(inside)
    }
}

impl Pattern for &Regex {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        // Find initial matches
        let matches = self
            .find_iter(inside)
            .map(|m| ((m.start(), m.end()), true))
            .collect::<Vec<_>>();

        // Then add missing splits inbetween
        let mut start_offset = 0;
        let mut splits = matches
            .into_iter()
            .flat_map(|((start, end), flag)| {
                let mut splits = vec![];
                if start_offset < start {
                    splits.push(((start_offset, start), false));
                }
                splits.push(((start, end), flag));
                start_offset = end;

                splits
            })
            .collect::<Vec<_>>();
        if let Some(((_, end), _)) = splits.iter().last().copied() {
            if end < inside.len() {
                splits.push(((end, inside.len()), false));
            }
        }

        Ok(splits)
    }
}

impl<F> Pattern for F
where
    F: Fn(char) -> bool,
{
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        let mut last_offset = 0;
        let mut last_seen = 0;

        let mut matches = inside
            .chars()
            .enumerate()
            .flat_map(|(i, c)| {
                last_seen = i;
                if self(c) {
                    let mut events = Vec::with_capacity(2);
                    if last_offset < i {
                        // We need to emit what was before this match
                        events.push(((last_offset, i), false));
                    }
                    events.push(((i, i + 1), true));
                    last_offset = i + 1;
                    events
                } else {
                    vec![]
                }
            })
            .collect::<Vec<_>>();

        // Do not forget the last potential split
        if last_seen >= last_offset {
            matches.push(((last_offset, last_seen + 1), false));
        }

        Ok(matches)
    }
}

/// Invert the `is_match` flags for the wrapped Pattern. This is usefull
/// for example when we use a regex that matches words instead of a delimiter,
/// and we want to match the delimiter.
pub struct Invert<P: Pattern>(pub P);
impl<P: Pattern> Pattern for Invert<P> {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        Ok(self
            .0
            .find_matches(inside)?
            .into_iter()
            .map(|(offsets, flag)| (offsets, !flag))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use regex::Regex;

    macro_rules! do_test {
        ($inside: expr, $pattern: expr => @ERROR) => {
            assert!($pattern.find_matches($inside).is_err());
        };
        ($inside: expr, $pattern: expr => $result: expr) => {
            assert_eq!($pattern.find_matches($inside).unwrap(), $result);
            assert_eq!(
                Invert($pattern).find_matches($inside).unwrap(),
                $result
                    .into_iter()
                    .map(|(offsets, flag)| (offsets, !flag))
                    .collect::<Vec<_>>()
            );
        };
    }

    #[test]
    fn char() {
        do_test!("aba", 'a' => vec![((0, 1), true), ((1, 2), false), ((2, 3), true)]);
        do_test!("bbbba", 'a' => vec![((0, 5), false), ((5, 6), true)]);
        do_test!("aabbb", 'a' => vec![((0, 1), true), ((1, 2), true), ((2, 5), false)]);
    }

    #[test]
    fn str() {
        do_test!("aba", "a" => vec![((0, 1), true), ((1, 2), false), ((2, 3), true)]);
        do_test!("bbbba", "a" => vec![((0, 5), false), ((5, 6), true)]);
        do_test!("aabbb", "a" => vec![((0, 1), true), ((1, 2), true), ((2, 5), false)]);
        do_test!("aabbb", "ab" => vec![((0, 1), false), ((1, 3), true), ((3, 5), false)]);
        do_test!("aabbab", "ab" =>
            vec![((0, 1), false), ((1, 3), true), ((3, 4), false), ((4, 6), true)]
        );
    }

    #[test]
    fn functions() {
        let is_b = |c| c == 'b';
        do_test!("aba", is_b => vec![((0, 1), false), ((1, 2), true), ((2, 3), false)]);
        do_test!("aaaab", is_b => vec![((0, 5), false), ((5, 6), true)]);
        do_test!("bbaaa", is_b => vec![((0, 1), true), ((1, 2), true), ((2, 5), false)]);
        do_test!("bbaaa", is_b => vec![((0, 1), false), ((1, 3), true), ((3, 5), false)]);
    }

    #[test]
    fn regex() {
        let is_whitespace = Regex::new(r"\s+").unwrap();
        do_test!("a   b", &is_whitespace => vec![((0, 1), false), ((1, 4), true), ((4, 5), false)]);
        do_test!("   a   b   ", &is_whitespace =>
            vec![((0, 4), true), ((4, 5), false), ((5, 8), true), ((8, 9), false), ((9, 12), true)]
        );
    }
}
