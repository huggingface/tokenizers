use crate::{Offsets, Result};
use bitcannon::literal::Literal;

/// Something to search for in a text: a [`Literal`] or a [`SysRegex`](super::SysRegex).
///
/// Callers read the matches through [`matches()`] or [`find_matches()`]. These two functions check
/// that the matches come in order without overlaps.
pub(super) trait Pattern {
    /// The byte range of every match in `inside`, left to right, as the matcher reports them.
    fn scan<'t>(&'t self, inside: &'t str) -> impl Iterator<Item = Result<Offsets>>;
}

/// The byte range of every match of `pattern` in `inside`, in order and without overlaps.
///
/// Errors on a match that starts before the previous one ended.
pub(super) fn matches<'t>(
    pattern: &'t impl Pattern,
    inside: &'t str,
) -> impl Iterator<Item = Result<Offsets>> {
    Ordered {
        matches: pattern.scan(inside),
        previous_end: 0,
    }
}

/// Every piece of `inside`, in order and covering all of it, flagged with whether `pattern` matched
/// it.
pub(super) fn find_matches(pattern: &impl Pattern, inside: &str) -> Result<Vec<(Offsets, bool)>> {
    if inside.is_empty() {
        return Ok(vec![((0, 0), false)]);
    }

    let mut previous_end = 0;
    let mut splits = Vec::with_capacity(inside.len());
    for found in matches(pattern, inside) {
        let (start, end) = found?;
        if previous_end != start {
            splits.push(((previous_end, start), false));
        }
        splits.push(((start, end), true));
        previous_end = end;
    }
    if previous_end != inside.len() {
        splits.push(((previous_end, inside.len()), false))
    }
    Ok(splits)
}

/// The matches of a [`Pattern`], checked to come in order without overlaps.
///
/// A `\K` inside a regex look-behind lets `fancy_regex` start a match before the previous one
/// ended. The splits built from such matches would be reversed or repeat text.
struct Ordered<I> {
    matches: I,
    previous_end: usize,
}

impl<I: Iterator<Item = Result<Offsets>>> Iterator for Ordered<I> {
    type Item = Result<Offsets>;

    fn next(&mut self) -> Option<Self::Item> {
        let (start, end) = match self.matches.next()? {
            Ok(found) => found,
            Err(error) => return Some(Err(error)),
        };
        if start < self.previous_end {
            return Some(Err(format!(
                "the pattern matched at byte {start}, inside the previous match that ended at byte {}. \
                 A `\\K` inside a regex look-behind does this, and overlapping matches cannot split text",
                self.previous_end,
            )
            .into()));
        }
        self.previous_end = end;
        Some(Ok((start, end)))
    }
}

/// A plain string needs no regex engine: [`Literal`] scans the bytes.
impl Pattern for Literal {
    fn scan<'t>(&'t self, inside: &'t str) -> impl Iterator<Item = Result<Offsets>> {
        let width = self.pattern().len();
        Literal::matches(self, inside.as_bytes()).map(move |start| Ok((start, start + width)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "fancy-regex")]
    use crate::tokenizer::pattern::SysRegex;

    #[test]
    fn literal() {
        let a = Literal::new(b"a").unwrap();
        assert_eq!(
            find_matches(&a, "aba").unwrap(),
            [((0, 1), true), ((1, 2), false), ((2, 3), true)]
        );
        assert_eq!(
            find_matches(&a, "bbbba").unwrap(),
            [((0, 4), false), ((4, 5), true)]
        );
        assert_eq!(
            find_matches(&a, "aabbb").unwrap(),
            [((0, 1), true), ((1, 2), true), ((2, 5), false)]
        );
        assert_eq!(find_matches(&a, "").unwrap(), [((0, 0), false)]);

        let ab = Literal::new(b"ab").unwrap();
        assert_eq!(
            find_matches(&ab, "aabbab").unwrap(),
            [
                ((0, 1), false),
                ((1, 3), true),
                ((3, 4), false),
                ((4, 6), true)
            ]
        );
        assert_eq!(
            find_matches(&Literal::new(b"b").unwrap(), "aaa").unwrap(),
            [((0, 3), false)]
        );

        // Offsets are byte offsets, over the whole input a multi-byte pattern covers.
        let metaspace = Literal::new("▁".as_bytes()).unwrap();
        assert_eq!(
            find_matches(&metaspace, "a▁b").unwrap(),
            [((0, 1), false), ((1, 4), true), ((4, 5), false)]
        );
    }

    #[test]
    #[cfg(feature = "fancy-regex")] // needs a system-regex backend
    fn sys_regex() {
        let is_whitespace = SysRegex::new(r"\s+").unwrap();
        assert_eq!(
            find_matches(&is_whitespace, "   a   b   ").unwrap(),
            [
                ((0, 3), true),
                ((3, 4), false),
                ((4, 7), true),
                ((7, 8), false),
                ((8, 11), true)
            ]
        );
        assert_eq!(find_matches(&is_whitespace, "").unwrap(), [((0, 0), false)]);
        assert_eq!(
            find_matches(&is_whitespace, "𝔾𝕠𝕠𝕕 𝕞𝕠𝕣𝕟𝕚𝕟𝕘").unwrap(),
            [((0, 16), false), ((16, 17), true), ((17, 45), false)]
        );
        assert_eq!(
            find_matches(&is_whitespace, "aaa").unwrap(),
            [((0, 3), false)]
        );
    }

    #[test]
    #[cfg(feature = "fancy-regex")] // needs a system-regex backend
    fn refuses_a_match_overlapping_the_previous_one() {
        // `\K` inside a look-behind moves a match start back into the previous match.
        let regex = SysRegex::new(r"(?<=\K.).|.").unwrap();
        assert!(find_matches(&regex, "Hello").is_err());
    }
}
