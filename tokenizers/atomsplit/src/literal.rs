//! Searching for a literal string, for pre-tokenizers that cut on one exact character.
//!
//! The rest of the crate works off atom tags: one SIMD pass gives every character a class, and the
//! FSMs cut where the class changes ([`crate::fsm`]).
//!
//! The atom classification is unnecessary for pre-tokenizers splitting on an exact character or literal string:
//! We can use a simpler byte search looking at 16 bytes at a time.

use memchr::memmem;
use std::fmt;

/// The pattern handed to [`Literal::new`] was empty.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmptyPattern;

impl fmt::Display for EmptyPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("an empty pattern matches everywhere")
    }
}

impl std::error::Error for EmptyPattern {}

/// A literal string to split on.
///
/// The finder is boxed because it is large — a few hundred bytes of prefilter state on x86_64 — and
/// callers store it inside enums whose other variants are tiny.
#[derive(Debug, Clone)]
pub struct Literal {
    finder: Box<memmem::Finder<'static>>,
}

impl Literal {
    /// # Errors
    /// If `pattern` is empty, which would match everywhere.
    pub fn new(pattern: &[u8]) -> Result<Self, EmptyPattern> {
        if pattern.is_empty() {
            return Err(EmptyPattern);
        }
        Ok(Self {
            finder: Box::new(memmem::Finder::new(pattern).into_owned()),
        })
    }

    /// The string being searched for.
    #[must_use]
    pub fn pattern(&self) -> &[u8] {
        self.finder.needle()
    }

    /// Byte offset of every match, left to right. Matches never overlap, so `"aa"` is found once in
    /// `"aaa"` — the same matches a regex engine would report.
    pub fn matches<'t>(&'t self, text: &'t [u8]) -> impl Iterator<Item = usize> + 't {
        self.finder.find_iter(text)
    }
}
