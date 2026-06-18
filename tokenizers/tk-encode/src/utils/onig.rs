use crate::tokenizer::pattern::Pattern;
use crate::{Offsets, Result};
use onig::Regex;
use std::error::Error;

#[derive(Debug)]
pub struct SysRegex {
    regex: Regex,
}

impl SysRegex {
    pub fn find_iter<'r, 't>(&'r self, inside: &'t str) -> onig::FindMatches<'r, 't> {
        self.regex.find_iter(inside)
    }

    pub fn new(
        regex_str: &str,
    ) -> std::result::Result<Self, Box<dyn Error + Send + Sync + 'static>> {
        Ok(Self {
            regex: Regex::new(regex_str)?,
        })
    }
}

impl Pattern for &Regex {
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
