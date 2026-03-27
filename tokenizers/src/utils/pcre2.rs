//! PCRE2 regex engine backend for tokenizers.
//!
//! This module provides a PCRE2-based backend as an alternative to the default Oniguruma regex engine.
//!
//! # Features
//!
//! Enable the `pcre2` feature to use this backend instead of the default Oniguruma engine:
//!
//! ```toml
//! [dependencies]
//! tokenizers = { version = "...", features = ["pcre2"] }
//! ```
//!
//! # Performance
//!
//! PCRE2 with JIT can offer ~16% faster encoding compared to the default
//! Oniguruma backend for models using BPE pre-tokenization.
//!
//!
//! # Build
//!
//! The underlying `pcre2-sys` crate will use a system-installed libpcre2 via pkg-config if
//! available, otherwise it builds PCRE2 from bundled source automatically. No external
//! dependencies are required.
//!

use pcre2::bytes::RegexBuilder;

#[derive(Debug)]
pub struct SysRegex {
    regex: pcre2::bytes::Regex,
}

impl SysRegex {
    pub fn new(
        regex_str: &str,
    ) -> std::result::Result<Self, Box<dyn std::error::Error + Send + Sync + 'static>> {
        let regex = RegexBuilder::new()
            .jit_if_available(true)
            .utf(true)
            .ucp(true)
            .build(regex_str)?;
        Ok(Self { regex })
    }

    pub fn find_iter<'r, 't>(&'r self, inside: &'t str) -> Matches<'r, 't> {
        Matches(self.regex.find_iter(inside.as_bytes()))
    }
}

pub struct Matches<'r, 't>(pcre2::bytes::Matches<'r, 't>);

impl Iterator for Matches<'_, '_> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        match self.0.next() {
            Some(Ok(mat)) => Some((mat.start(), mat.end())),
            None | Some(Err(_)) => None,
        }
    }
}
