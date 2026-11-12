use super::matcher::Pattern;
use crate::{Offsets, Result};
use fancy_regex::Regex;

#[derive(Debug)]
pub struct SysRegex {
    regex: Regex,
}

impl SysRegex {
    pub fn new(regex_str: &str) -> Result<Self> {
        Ok(Self {
            regex: Regex::new(regex_str)?,
        })
    }
}

impl Pattern for SysRegex {
    fn scan<'t>(&'t self, inside: &'t str) -> impl Iterator<Item = Result<Offsets>> {
        self.regex.find_iter(inside).map(|found| {
            let found = found?;
            Ok((found.start(), found.end()))
        })
    }
}
