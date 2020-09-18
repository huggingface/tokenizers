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
        if self.is_empty() {
            // If we try to find the matches with an empty string, just don't match anything
            return Ok(vec![((0, inside.chars().count()), false)]);
        }

        let re = Regex::new(&regex::escape(self))?;
        (&re).find_matches(inside)
    }
}

impl Pattern for &String {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        let s: &str = self;
        s.find_matches(inside)
    }
}

impl Pattern for &Regex {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        if inside.is_empty() {
            return Ok(vec![((0, 0), false)]);
        }

        let mut prev = 0;
        let mut splits = Vec::with_capacity(inside.len());
        for m in self.find_iter(inside) {
            if prev != m.start() {
                splits.push(((prev, m.start()), false));
            }
            splits.push(((m.start(), m.end()), true));
            prev = m.end();
        }
        if prev != inside.len() {
            splits.push(((prev, inside.len()), false))
        }
        Ok(splits)
    }
}

impl Pattern for &onig::Regex {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        if inside.is_empty() {
            return Ok(vec![((0, 0), false)]);
        }

        let mut prev = 0;
        let mut splits = Vec::with_capacity(inside.len());
        for (start, end) in self.find_iter(inside) {
            if prev != start {
                splits.push(((prev, start), false));
            }
            splits.push(((start, end), true));
            prev = end;
        }
        if prev != inside.len() {
            splits.push(((prev, inside.len()), false))
        }
        Ok(splits)
    }
}

impl<F> Pattern for F
where
    F: Fn(char) -> bool,
{
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        if inside.is_empty() {
            return Ok(vec![((0, 0), false)]);
        }

        let mut last_offset = 0;
        let mut last_seen = 0;

        let mut matches = inside
            .char_indices()
            .flat_map(|(b, c)| {
                last_seen = b + c.len_utf8();
                if self(c) {
                    let mut events = Vec::with_capacity(2);
                    if last_offset < b {
                        // We need to emit what was before this match
                        events.push(((last_offset, b), false));
                    }
                    events.push(((b, b + c.len_utf8()), true));
                    last_offset = b + c.len_utf8();
                    events
                } else {
                    vec![]
                }
            })
            .collect::<Vec<_>>();

        // Do not forget the last potential split
        if last_seen > last_offset {
            matches.push(((last_offset, last_seen), false));
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
                    .map(|v: (Offsets, bool)| (v.0, !v.1))
                    .collect::<Vec<_>>()
            );
        };
    }

    #[test]
    fn char() {
        do_test!("aba", 'a' => vec![((0, 1), true), ((1, 2), false), ((2, 3), true)]);
        do_test!("bbbba", 'a' => vec![((0, 4), false), ((4, 5), true)]);
        do_test!("aabbb", 'a' => vec![((0, 1), true), ((1, 2), true), ((2, 5), false)]);
        do_test!("", 'a' => vec![((0, 0), false)]);
        do_test!("aaa", 'b' => vec![((0, 3), false)]);
    }

    #[test]
    fn str() {
        do_test!("aba", "a" => vec![((0, 1), true), ((1, 2), false), ((2, 3), true)]);
        do_test!("bbbba", "a" => vec![((0, 4), false), ((4, 5), true)]);
        do_test!("aabbb", "a" => vec![((0, 1), true), ((1, 2), true), ((2, 5), false)]);
        do_test!("aabbb", "ab" => vec![((0, 1), false), ((1, 3), true), ((3, 5), false)]);
        do_test!("aabbab", "ab" =>
            vec![((0, 1), false), ((1, 3), true), ((3, 4), false), ((4, 6), true)]
        );
        do_test!("", "" => vec![((0, 0), false)]);
        do_test!("aaa", "" => vec![((0, 3), false)]);
        do_test!("aaa", "b" => vec![((0, 3), false)]);
    }

    #[test]
    fn functions() {
        let is_b = |c| c == 'b';
        do_test!("aba", is_b => vec![((0, 1), false), ((1, 2), true), ((2, 3), false)]);
        do_test!("aaaab", is_b => vec![((0, 4), false), ((4, 5), true)]);
        do_test!("bbaaa", is_b => vec![((0, 1), true), ((1, 2), true), ((2, 5), false)]);
        do_test!("", is_b => vec![((0, 0), false)]);
        do_test!("aaa", is_b => vec![((0, 3), false)]);
    }

    #[test]
    fn regex() {
        let is_whitespace = Regex::new(r"\s+").unwrap();
        do_test!("a   b", &is_whitespace => vec![((0, 1), false), ((1, 4), true), ((4, 5), false)]);
        do_test!("   a   b   ", &is_whitespace =>
            vec![((0, 3), true), ((3, 4), false), ((4, 7), true), ((7, 8), false), ((8, 11), true)]
        );
        do_test!("", &is_whitespace => vec![((0, 0), false)]);
        do_test!("𝔾𝕠𝕠𝕕 𝕞𝕠𝕣𝕟𝕚𝕟𝕘", &is_whitespace =>
            vec![((0, 16), false), ((16, 17), true), ((17, 45), false)]
        );
        do_test!("aaa", &is_whitespace => vec![((0, 3), false)]);
    }

    #[test]
    fn onig_regex() {
        let is_whitespace = onig::Regex::new(r"\s+").unwrap();
        do_test!("a   b", &is_whitespace => vec![((0, 1), false), ((1, 4), true), ((4, 5), false)]);
        do_test!("   a   b   ", &is_whitespace =>
            vec![((0, 3), true), ((3, 4), false), ((4, 7), true), ((7, 8), false), ((8, 11), true)]
        );
        do_test!("", &is_whitespace => vec![((0, 0), false)]);
        do_test!("𝔾𝕠𝕠𝕕 𝕞𝕠𝕣𝕟𝕚𝕟𝕘", &is_whitespace =>
            vec![((0, 16), false), ((16, 17), true), ((17, 45), false)]
        );
        do_test!("aaa", &is_whitespace => vec![((0, 3), false)]);
    }
}
