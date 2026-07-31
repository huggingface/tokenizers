use crate::utils::SysRegex;
use crate::{Offsets, Result};
use atomsplit::literal::Literal;
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

/// Searching for a plain string, does not need a regex engine: [`Literal`] scans the bytes.
/// The batch scan ([`Literal::matches_into`]) finds every match in one pass, and knowing the
/// count first sizes `splits` exactly: each match adds itself and at most one gap before it,
/// plus the final gap.
impl Pattern for &Literal {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        if inside.is_empty() {
            return Ok(vec![((0, 0), false)]);
        }

        let width = self.pattern().len();
        let mut positions = vec![0u32; inside.len() / width + 4];
        let count = self.matches_into(inside.as_bytes(), &mut positions);

        let mut prev = 0;
        let mut splits = Vec::with_capacity(2 * count + 1);
        for &start in &positions[..count] {
            let (start, end) = (start as usize, start as usize + width);
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

/// A plain string is one [`Literal`], built here because the pattern only lives for this call. Prefer
/// keeping a [`Literal`] around when the same pattern is searched for repeatedly.
impl Pattern for &str {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        match Literal::new(self.as_bytes()) {
            Ok(literal) => (&literal).find_matches(inside),
            // An empty pattern would match everywhere, so it matches nowhere instead.
            Err(_) => Ok(vec![((0, inside.len()), false)]),
        }
    }
}

impl Pattern for &String {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        let s: &str = self;
        s.find_matches(inside)
    }
}

impl Pattern for &SysRegex {
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

/// Invert the `is_match` flags for the wrapped Pattern. This is useful
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
    fn literal() {
        let a = Literal::new(b"a").unwrap();
        do_test!("aba", &a => vec![((0, 1), true), ((1, 2), false), ((2, 3), true)]);
        do_test!("bbbba", &a => vec![((0, 4), false), ((4, 5), true)]);
        do_test!("aabbb", &a => vec![((0, 1), true), ((1, 2), true), ((2, 5), false)]);
        do_test!("", &a => vec![((0, 0), false)]);

        let ab = Literal::new(b"ab").unwrap();
        do_test!("aabbb", &ab => vec![((0, 1), false), ((1, 3), true), ((3, 5), false)]);
        do_test!("aabbab", &ab =>
            vec![((0, 1), false), ((1, 3), true), ((3, 4), false), ((4, 6), true)]
        );

        let b = Literal::new(b"b").unwrap();
        do_test!("aaa", &b => vec![((0, 3), false)]);

        // Offsets are byte offsets, over the whole input a multi-byte pattern covers.
        let metaspace = Literal::new("▁".as_bytes()).unwrap();
        do_test!("a▁b", &metaspace => vec![((0, 1), false), ((1, 4), true), ((4, 5), false)]);

        // An empty pattern would match everywhere, so there is no `Literal` for it.
        assert!(Literal::new(b"").is_err());
    }

    #[test]
    fn str() {
        do_test!("aba", "a" => vec![((0, 1), true), ((1, 2), false), ((2, 3), true)]);
        do_test!("aabbab", "ab" =>
            vec![((0, 1), false), ((1, 3), true), ((3, 4), false), ((4, 6), true)]
        );
        do_test!("aaa", "b" => vec![((0, 3), false)]);
        do_test!("", "" => vec![((0, 0), false)]);
        // An empty pattern would match everywhere, so it matches nowhere instead.
        do_test!("aaa", "" => vec![((0, 3), false)]);

        let owned = String::from("ab");
        do_test!("aabbb", &owned => vec![((0, 1), false), ((1, 3), true), ((3, 5), false)]);
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
    #[cfg(feature = "fancy-regex")] // needs a system-regex backend
    fn sys_regex() {
        let is_whitespace = SysRegex::new(r"\s+").unwrap();
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
