use fancy_regex::Regex;
use std::error::Error;

#[derive(Debug)]
pub struct SysRegex {
    regex: Regex,
}

impl SysRegex {
    pub fn find_iter<'r, 't>(&'r self, inside: &'t str) -> Matches<'r, 't> {
        Matches(self.regex.find_iter(inside))
    }

    pub fn new(regex_str: &str) -> Result<Self, Box<dyn Error + Send + Sync + 'static>> {
        Ok(Self {
            regex: Regex::new(regex_str)?,
        })
    }
}

pub struct Matches<'r, 't>(fancy_regex::Matches<'r, 't>);

impl<'r, 't> Iterator for Matches<'r, 't> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        match self.0.next() {
            Some(Ok(mat)) => Some((mat.start(), mat.end())),
            // stop if an error is encountered
            None | Some(Err(_)) => None,
        }
    }
}
