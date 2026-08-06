//! Searching for a literal string, for pre-tokenizers that cut on one exact character.
//!
//! The rest of the crate works off atom tags: one SIMD pass gives every character a class, and the
//! FSMs cut where the class changes ([`crate::fsm`]).
//!
//! The atom classification is unnecessary for pre-tokenizers splitting on an exact character or literal string:
//! We can use a simpler byte search looking at 16 bytes at a time.

use std::{fmt, u32};
use wide::{self, u8x32};

/// The pattern handed to [`Literal::new`] cannot be searched for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidPattern {
    /// An empty pattern would match everywhere.
    Empty,
    /// Longer than [`Literal::MAX_PATTERN_LEN`] bytes.
    TooLong,
}

impl fmt::Display for InvalidPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("an empty pattern matches everywhere"),
            Self::TooLong => write!(
                f,
                "patterns longer than {} bytes are not supported",
                Literal::MAX_PATTERN_LEN
            ),
        }
    }
}

impl std::error::Error for InvalidPattern {}

/// A literal string to split on, at most [`Literal::MAX_PATTERN_LEN`] bytes.
#[derive(Debug, Clone)]
pub struct Literal {
    pattern: [u8; Self::MAX_PATTERN_LEN],
    pattern_len: u8,
}

impl Literal {
    /// Longest supported pattern: the matcher compares one 32-byte block at a time, so a longer
    /// pattern could never match inside a block.
    pub const MAX_PATTERN_LEN: usize = 32;

    /// # Errors
    /// If `pattern` is empty (it would match everywhere) or longer than
    /// [`Literal::MAX_PATTERN_LEN`] bytes.
    pub fn new(pattern: &[u8]) -> Result<Self, InvalidPattern> {
        if pattern.is_empty() {
            return Err(InvalidPattern::Empty);
        }
        if pattern.len() > Self::MAX_PATTERN_LEN {
            return Err(InvalidPattern::TooLong);
        }
        let mut buf = [0u8; Self::MAX_PATTERN_LEN];
        buf[..pattern.len()].copy_from_slice(pattern);
        Ok(Self {
            pattern: buf,
            pattern_len: pattern.len() as u8,
        })
    }

    /// The string being searched for.
    #[must_use]
    pub fn pattern(&self) -> &[u8] {
        &self.pattern[..usize::from(self.pattern_len)]
    }

    /// Byte offset of every match, left to right. Matches never overlap, so `"aa"` is found once in
    /// `"aaa"` — the same matches a regex engine would report.
    pub fn matches<'t>(&'t self, text: &'t [u8]) -> impl Iterator<Item = usize> + 't {
        LiteralMatcher::new(text, self.pattern())
    }
}

struct LiteralMatcher<'a> {
    text: &'a [u8],
    needle: &'a [u8],
    needle_splats: Vec<u8x32>,
    equality_mask: u32,
    block_start: usize,
    next_block: usize,
    min_next_start: usize,
}

impl<'a> LiteralMatcher<'a> {
    pub fn new(text: &'a [u8], needle: &'a [u8]) -> Self {
        Self {
            text,
            needle,
            needle_splats: needle.iter().map(|&b| u8x32::splat(b)).collect(),
            equality_mask: 0,
            block_start: 0,
            next_block: 0,
            min_next_start: 0,
        }
    }

    pub fn load_more(&mut self) -> Option<usize> {
        if self.next_block >= self.text.len() {
            // We have read the whole text
            return None;
        }
        self.block_start = self.next_block;
        let remaining = &self.text[self.block_start..];
        let len_to_load = remaining.len().min(32);
        let mut padded_text = [0u8; 32];
        padded_text[..len_to_load].copy_from_slice(&remaining[..len_to_load]);
        let simd_bytes = u8x32::new(padded_text);
        let mut mask = u32::MAX;
        for (idx, &needle_splat) in self.needle_splats.iter().enumerate() {
            mask &= simd_bytes.simd_eq(needle_splat).to_bitmask() >> idx;
        }

        mask &= u32::MAX >> (self.needle.len() - 1);
        if len_to_load < 32 {
            let valid = len_to_load.saturating_sub(self.needle.len() - 1);
            mask &= (1u32 << valid) - 1
        }
        self.equality_mask = mask;
        self.next_block += 32 - (self.needle.len() - 1);
        Some(len_to_load)
    }

    pub fn next_match(&mut self) -> Option<usize> {
        loop {
            // Load more until a match or text has been entirely read
            while self.equality_mask == 0 {
                self.load_more()?;
            }
            let offset = self.equality_mask.trailing_zeros() as usize;
            self.equality_mask &= self.equality_mask - 1;
            let start = self.block_start + offset;
            if start >= self.min_next_start {
                self.min_next_start = start + self.needle.len();
                return Some(start);
            }
        }
    }
}

impl<'a> Iterator for LiteralMatcher<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_match()
    }
}
