//! Searching for a literal string, for pre-tokenizers that cut on one exact character.
//!
//! The rest of the crate works off atom tags: one SIMD pass gives every character a class, and the
//! FSMs cut where the class changes ([`crate::fsm`]).
//!
//! The atom classification is unnecessary for pre-tokenizers splitting on an exact character or literal string:
//! a plain byte search finds the same cuts. Two searches cover the two match densities:
//!
//! - [`Literal::matches`] iterates with [`memmem`], which is built for needles that are rare in
//!   the haystack: it skips ahead fast and restarts after every match.
//! - [`Literal::matches_into`] scans the whole text once and writes every offset into a caller
//!   buffer. Pre-tokenizer delimiters are the opposite of rare (running English text has a space
//!   about every six bytes), and at that density the per-match restarts dominate the iterator.
//!   The scan answers a whole block of text at once and turns the answers into offsets;
//!   `simd_literal.rs` explains how, step by step.

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

    /// The same offsets as [`Literal::matches`], written into `out[..count]` in one scan of
    /// `text`; returns the count. Use this when matches are frequent: the iterator restarts its
    /// search machinery after every match, the scan never stops.
    ///
    /// # Preconditions
    /// - `out.len() >= text.len() / pattern.len() + 4` (asserted). The division is the most
    ///   matches the text can hold; the `+ 4` is slack the SIMD path writes past the last match.
    /// - `text.len()` must fit in `u32` (asserted), so that every offset does too.
    pub fn matches_into(&self, text: &[u8], out: &mut [u32]) -> usize {
        assert!(
            out.len() >= text.len() / self.pattern().len() + 4,
            "matches_into needs out.len() >= text.len() / pattern.len() + 4"
        );
        assert!(
            u32::try_from(text.len()).is_ok(),
            "matches_into writes u32 offsets"
        );
        #[cfg(any(
            target_arch = "aarch64",
            target_arch = "x86_64",
            all(target_arch = "wasm32", target_feature = "simd128")
        ))]
        if self.scannable() {
            match *self.pattern() {
                [a] => return crate::simd_literal::matches_into([a], text, out),
                [a, b] => return crate::simd_literal::matches_into([a, b], text, out),
                [a, b, c] => return crate::simd_literal::matches_into([a, b, c], text, out),
                _ => unreachable!("scannable patterns are one to three bytes"),
            }
        }
        let mut count = 0;
        for position in self.finder.find_iter(text) {
            out[count] = position as u32;
            count += 1;
        }
        count
    }

    /// How many matches [`Literal::matches`] would report. Counting runs the compare step
    /// without producing any offsets, so it is several times faster than a full scan; count
    /// first to size an output exactly, then fill it with a [`Literal::for_each_match`] pass.
    #[must_use]
    pub fn count_matches(&self, text: &[u8]) -> usize {
        #[cfg(any(
            target_arch = "aarch64",
            target_arch = "x86_64",
            all(target_arch = "wasm32", target_feature = "simd128")
        ))]
        if self.scannable() {
            match *self.pattern() {
                [a] => return crate::simd_literal::count_matches([a], text),
                [a, b] => return crate::simd_literal::count_matches([a, b], text),
                [a, b, c] => return crate::simd_literal::count_matches([a, b, c], text),
                _ => unreachable!("scannable patterns are one to three bytes"),
            }
        }
        self.finder.find_iter(text).count()
    }

    /// Calls `on_match` with the byte offset of every match, left to right: the same offsets
    /// as [`Literal::matches`] at batch-scan speed, with no buffer for the caller to provide.
    /// The scan streams through a small stack window, so its footprint stays flat however
    /// long the text is.
    pub fn for_each_match(&self, text: &[u8], mut on_match: impl FnMut(usize)) {
        if !self.scannable() {
            for position in self.finder.find_iter(text) {
                on_match(position);
            }
            return;
        }
        self.scan_windows(text, |base, matches| {
            for &position in matches {
                on_match(base + position as usize);
            }
        });
    }

    /// Whether the batch scan covers this pattern. The scan emits every position where the
    /// pattern matches, with no ordering check between them; that is only the non-overlapping
    /// match list when the pattern cannot overlap itself, which one byte comparison per
    /// length decides: a two-byte pattern overlaps itself when both bytes are equal, a
    /// three-byte one when last equals first. Longer or self-overlapping patterns search
    /// through the [`memmem`] engine instead.
    #[cfg(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    fn scannable(&self) -> bool {
        match *self.pattern() {
            [_] => true,
            [a, b] => a != b,
            [a, _, c] => a != c,
            _ => false,
        }
    }

    /// Without a SIMD kernel there is no batch scan; everything searches through [`memmem`].
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    fn scannable(&self) -> bool {
        false
    }

    /// Streams the batch scan window by window: [`Literal::matches_into`] fills a stack
    /// buffer for one stretch of text, `on_window` consumes it (offsets are relative to the
    /// window's `base`), and the next window starts where a match could no longer fit in the
    /// previous one, so nothing at a window edge is missed or reported twice.
    fn scan_windows(&self, text: &[u8], mut on_window: impl FnMut(usize, &[u32])) {
        // 4KB of stack; a window of (1024 - 4) * pattern.len() bytes fills it exactly.
        let mut buffer = [0u32; 1024];
        let width = self.pattern().len();
        let window = (buffer.len() - 4) * width;
        let mut base = 0;
        loop {
            let end = usize::min(base + window, text.len());
            let count = self.matches_into(&text[base..end], &mut buffer);
            on_window(base, &buffer[..count]);
            if end == text.len() {
                return;
            }
            // A match crossing `end` was not reported; the next window re-covers its
            // possible starts. Matches of a scannable pattern never overlap, so the reports
            // stay disjoint.
            base = end - (width - 1);
        }
    }
}
