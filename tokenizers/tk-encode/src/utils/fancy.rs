use fancy_regex::Regex;
use std::error::Error;

#[derive(Debug)]
pub struct SysRegex {
    regex: Regex,
}

impl SysRegex {
    pub(crate) fn find_iter<'r, 't>(&'r self, inside: &'t str) -> Matches<'r, 't> {
        Matches {
            matches: self.regex.find_iter(inside),
            previous_end: 0,
        }
    }

    pub fn new(regex_str: &str) -> Result<Self, Box<dyn Error + Send + Sync + 'static>> {
        Ok(Self {
            regex: Regex::new(regex_str)?,
        })
    }
}

/// The matches of a [`SysRegex`], checked to come in order without overlaps.
///
/// `Split` builds spans from these matches and its `unsafe impl PreTokenize` relies on this check.
pub struct Matches<'r, 't> {
    matches: fancy_regex::Matches<'r, 't, str>,
    previous_end: usize,
}

impl Iterator for Matches<'_, '_> {
    type Item = crate::Result<(usize, usize)>;

    fn next(&mut self) -> Option<Self::Item> {
        // stop if an error is encountered
        let found = self.matches.next()?.ok()?;
        let (start, end) = (found.start(), found.end());
        if start < self.previous_end || end < start {
            return Some(Err(format!(
                "the regex matched bytes {start}..{end}, a range that is reversed or overlaps the \
                 previous match, which ended at byte {}.",
                self.previous_end,
            )
            .into()));
        }
        self.previous_end = end;
        Some(Ok((start, end)))
    }
}
