use crate::tokenizer::pattern::Pattern;
use crate::Offsets;
use std::error::Error;

/// PCRE2-backed regex with JIT compilation for fast matching.
///
/// Uses `pcre2::bytes::Regex` with UTF-8 and Unicode property support enabled.
/// JIT compilation is requested when available, compiling the pattern to native
/// machine code for significantly faster matching.
pub struct SysRegex {
    regex: pcre2::bytes::Regex,
}

impl std::fmt::Debug for SysRegex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SysRegex(pcre2)")
    }
}

impl SysRegex {
    pub fn find_iter<'r, 't>(&'r self, inside: &'t str) -> Matches<'r, 't> {
        Matches {
            regex: &self.regex,
            text: inside.as_bytes(),
            offset: 0,
        }
    }

    pub fn new(regex_str: &str) -> Result<Self, Box<dyn Error + Send + Sync + 'static>> {
        let regex = pcre2::bytes::RegexBuilder::new()
            .utf(true)
            .ucp(true)
            .jit_if_available(true)
            .build(regex_str)?;
        Ok(Self { regex })
    }
}

/// Iterator over PCRE2 regex matches, yielding `(start, end)` byte offsets.
pub struct Matches<'r, 't> {
    regex: &'r pcre2::bytes::Regex,
    text: &'t [u8],
    offset: usize,
}

impl Iterator for Matches<'_, '_> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset > self.text.len() {
            return None;
        }
        match self.regex.find_at(self.text, self.offset) {
            Ok(Some(m)) => {
                let start = m.start();
                let end = m.end();
                // Advance past this match (handle zero-length matches)
                if end == self.offset {
                    self.offset += 1;
                    // Skip forward to next valid UTF-8 boundary
                    while self.offset < self.text.len()
                        && (self.text[self.offset] & 0xC0) == 0x80
                    {
                        self.offset += 1;
                    }
                } else {
                    self.offset = end;
                }
                Some((start, end))
            }
            Ok(None) | Err(_) => None,
        }
    }
}

impl Pattern for &pcre2::bytes::Regex {
    fn find_matches(
        &self,
        inside: &str,
    ) -> Result<Vec<(Offsets, bool)>, Box<dyn Error + Send + Sync + 'static>> {
        if inside.is_empty() {
            return Ok(vec![((0, 0), false)]);
        }

        let mut prev = 0;
        let mut splits = Vec::with_capacity(inside.len());
        let mut offset = 0;
        let text = inside.as_bytes();
        while offset <= text.len() {
            match self.find_at(text, offset) {
                Ok(Some(m)) => {
                    let start = m.start();
                    let end = m.end();
                    if prev != start {
                        splits.push(((prev, start), false));
                    }
                    splits.push(((start, end), true));
                    prev = end;
                    if end == offset {
                        offset += 1;
                        while offset < text.len() && (text[offset] & 0xC0) == 0x80 {
                            offset += 1;
                        }
                    } else {
                        offset = end;
                    }
                }
                Ok(None) | Err(_) => break,
            }
        }
        if prev != inside.len() {
            splits.push(((prev, inside.len()), false));
        }
        Ok(splits)
    }
}
