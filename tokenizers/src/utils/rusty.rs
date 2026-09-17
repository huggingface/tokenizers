use crate::tokenizer::pattern::Pattern;
use crate::Offsets;
use rusty_expressions::{Encoding, MatchParam, Options, Regex, Syntax};
use std::error::Error;

/// `SysRegex` backed by `rusty_expressions` -- Oniguruma remade in pure Rust.
///
/// Same engine semantics as the `onig` backend, because it is a
/// reimplementation of Oniguruma gated differentially against `libonig`, but
/// with no C in the build: it compiles for `wasm32`, needs no `cc`, and is
/// measured about 3x faster than `libonig` on search.
#[derive(Debug)]
pub struct SysRegex {
    regex: Regex,
}

impl SysRegex {
    pub fn find_iter<'r, 't>(&'r self, inside: &'t str) -> Matches<'r, 't> {
        Matches {
            regex: &self.regex,
            hay: inside,
            pos: 0,
            done: false,
        }
    }

    pub fn new(regex_str: &str) -> Result<Self, Box<dyn Error + Send + Sync + 'static>> {
        let regex = Regex::new(
            regex_str.as_bytes(),
            Options::NONE,
            Encoding::UTF8,
            Syntax::ONIGURUMA,
        )
        .map_err(|e| -> Box<dyn Error + Send + Sync + 'static> { e.to_string().into() })?;
        Ok(Self { regex })
    }
}

/// Non-overlapping matches, left to right.
///
/// An empty match advances by one character so the iterator always makes
/// progress, matching what the `onig` backend yields.
pub struct Matches<'r, 't> {
    regex: &'r Regex,
    hay: &'t str,
    pos: usize,
    done: bool,
}

impl Iterator for Matches<'_, '_> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.done || self.pos > self.hay.len() {
            return None;
        }
        let bytes = self.hay.as_bytes();
        let found = self
            .regex
            .search_range_param(bytes, self.pos, bytes.len(), &MatchParam::default());
        let m = match found {
            Ok(Some(m)) => m,
            // Stop on mismatch, and on an engine limit -- the `fancy-regex`
            // backend stops on error too rather than propagating.
            _ => {
                self.done = true;
                return None;
            }
        };
        let r = m.range();
        self.pos = if r.end > r.start {
            r.end
        } else {
            // Empty match: step one whole character, never a byte, so we
            // cannot land inside a UTF-8 sequence.
            let mut n = r.end + 1;
            while n < self.hay.len() && !self.hay.is_char_boundary(n) {
                n += 1;
            }
            n
        };
        Some((r.start, r.end))
    }
}

impl Pattern for &Regex {
    fn find_matches(
        &self,
        inside: &str,
    ) -> Result<Vec<(Offsets, bool)>, Box<dyn Error + Send + Sync + 'static>> {
        if inside.is_empty() {
            return Ok(vec![((0, 0), false)]);
        }

        let wrapper = SysRegex {
            regex: (*self).clone(),
        };
        let mut prev = 0;
        let mut splits = Vec::with_capacity(inside.len());
        for (start, end) in wrapper.find_iter(inside) {
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
