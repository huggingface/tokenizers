use crate::Result;
use crate::utils::SysRegex;
use bitcannon::literal::Literal;

/// Pattern used by `Split` and `Replace` to find the spans to split on or replace.
pub(crate) trait Pattern {
    /// Slice the given string in a list of pattern match positions, with
    /// a boolean indicating whether this is a match or not.
    ///
    /// This method *must* cover the whole string in its outputs, with
    /// contiguous ordered slices.
    fn find_matches(&self, inside: &str) -> Result<Vec<((usize, usize), bool)>>;
}

/// Searching for a plain string, does not need a regex engine: [`Literal`] scans the bytes.
impl Pattern for &Literal {
    fn find_matches(&self, inside: &str) -> Result<Vec<((usize, usize), bool)>> {
        if inside.is_empty() {
            return Ok(vec![((0, 0), false)]);
        }

        let mut prev = 0;
        let mut splits = Vec::with_capacity(inside.len());
        for start in self.matches(inside.as_bytes()) {
            let end = start + self.pattern().len();
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

impl Pattern for &SysRegex {
    fn find_matches(&self, inside: &str) -> Result<Vec<((usize, usize), bool)>> {
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

/// Invert the `is_match` flags for the wrapped Pattern. This is useful
/// for example when we use a regex that matches words instead of a delimiter,
/// and we want to match the delimiter.
pub(crate) struct Invert<P: Pattern>(pub P);
impl<P: Pattern> Pattern for Invert<P> {
    fn find_matches(&self, inside: &str) -> Result<Vec<((usize, usize), bool)>> {
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
                    .map(|v: ((usize, usize), bool)| (v.0, !v.1))
                    .collect::<Vec<_>>()
            );
        };
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
