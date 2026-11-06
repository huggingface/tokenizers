use crate::pipeline;
use crate::tokenizer::Result;
use crate::utils::unicode::{MARKS_ONLY_IN_UNICODE_9, Unicode9IgnoredCombiningMarks};
use std::borrow::Cow;
use unicode_normalization::char::is_combining_mark;

/// Removes combining marks from the text
#[derive(Clone)]
pub struct StripAccents {
    // This is needed for backwards compatibility with Unicode 9
    unicode_17_accents: Unicode9IgnoredCombiningMarks,
}

impl Default for StripAccents {
    fn default() -> Self {
        Self::new()
    }
}

impl StripAccents {
    pub fn new() -> Self {
        Self {
            unicode_17_accents: Unicode9IgnoredCombiningMarks::new(),
        }
    }

    fn is_combining_mark(&self, c: char) -> bool {
        if is_combining_mark(c) {
            // Keep backwards compatibility with Unicode 9:
            // Don't strip combining marks that were not combining in
            // Unicode 9 but are in Unicode 17
            !self.unicode_17_accents.contains(c)
        } else {
            // Keep backwards compatibility with Unicode 9:
            // Do strip combining marks that were combining in
            // Unicode 9 but are not in Unicode 17
            MARKS_ONLY_IN_UNICODE_9.contains(&c)
        }
    }
}

impl std::fmt::Debug for StripAccents {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("StripAccents")
    }
}

impl pipeline::Normalizer for StripAccents {
    fn normalize<'a>(&self, input: &'a str, _offset: usize) -> Result<Cow<'a, str>> {
        if input.chars().any(|c| self.is_combining_mark(c)) {
            Ok(Cow::Owned(
                input
                    .chars()
                    .filter(|&c| !self.is_combining_mark(c))
                    .collect(),
            ))
        } else {
            Ok(Cow::Borrowed(input))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_accents() {
        let n = StripAccents::new();
        for (input, expected) in [
            // NFD output: the accent is its own char, U+0301.
            ("cafe\u{301}", "cafe"),
            ("a\u{30A}\u{327}", "a"),
            // Precomposed letters are not marks.
            ("café", "café"),
            // Unicode 9 backwards compatibility:
            // U+07FD NKO DANTAYALAN became a mark in Unicode 11.
            ("a\u{07FD}b", "a\u{07FD}b"),
            // U+1CF2 VEDIC SIGN ARDHAVISARGA stopped being a mark in Unicode 10.
            ("a\u{1CF2}b", "ab"),
        ] {
            assert_eq!(
                &*pipeline::Normalizer::normalize(&n, input, 0).unwrap(),
                expected,
                "input={input:?}"
            );
        }
    }

    #[test]
    fn strip_accents_borrows_text_without_marks() {
        let n = StripAccents::new();
        assert!(matches!(
            pipeline::Normalizer::normalize(&n, "café", 0).unwrap(),
            Cow::Borrowed(_)
        ));
    }
}
