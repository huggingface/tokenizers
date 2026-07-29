//! Searching for a literal string, for pre-tokenizers that cut on one exact character.
//!
//! The rest of the crate works off atom tags: one SIMD pass gives every character a class, and the
//! FSMs cut where the class changes ([`crate::fsm`]). A pre-tokenizer that cuts on an exact character
//! needs none of that, and is better off without it: a search skips 16 bytes at a time, while
//! classifying has to write a tag for every single one. On text where the character is rare that is
//! the difference between touching a twentieth of the bytes and all of them.
//!
//! Text is `&[u8]` here, and no decoding happens. UTF-8 is self-synchronizing: the bytes of one
//! character never appear inside another, so a match can only ever start where a character does.

use memchr::memmem;

/// A literal string to split on.
///
/// Build it once and keep it. The constructor works out which of the pattern's bytes are rarest in
/// ordinary text, and searching for those instead of the first byte is what makes it fast.
#[derive(Debug, Clone)]
pub struct Literal {
    finder: memmem::Finder<'static>,
}

impl Literal {
    /// # Panics
    /// If `pattern` is empty, which would match everywhere.
    #[must_use]
    pub fn new(pattern: &[u8]) -> Self {
        assert!(!pattern.is_empty(), "an empty pattern matches everywhere");
        Self {
            // `into_owned` copies the pattern in, so the caller's slice does not have to outlive this.
            finder: memmem::Finder::new(pattern).into_owned(),
        }
    }

    /// The string being searched for.
    #[must_use]
    pub fn pattern(&self) -> &[u8] {
        self.finder.needle()
    }

    /// Byte offset of every match, left to right. Matches never overlap, so `"aa"` is found once in
    /// `"aaa"` — the same matches a regex engine would report.
    ///
    /// Searching for the whole pattern is what makes a multi-byte one cheap *and* correct: `▁`
    /// (U+2581) is the three bytes `E2 96 81`, and `E2` also starts `—`, `“` and `…`. Looking for
    /// that first byte alone would stop on every one of those and then have to check the rest.
    pub fn matches<'t>(&'t self, text: &'t [u8]) -> impl Iterator<Item = usize> + 't {
        self.finder.find_iter(text)
    }
}
