//! Writes SentencePiece's `▁` into the text, where the spaces used to be.
//!
//! SentencePiece tokenizers have no token for a space: a space is folded into the word that follows
//! it and written `▁`. That character therefore marks where a word starts *and* separates words —
//! which is why the code below calls it the delimiter. In a tokenizer config, one `Metaspace`
//! pre-tokenizer does the whole job: it rewrites the text and cuts it into words. The pipeline keeps
//! those two apart — normalizers rewrite text, pre-tokenizers cut it — so a `Metaspace` is rebuilt
//! as this normalizer plus a [`Split`] on the delimiter. There is no `Metaspace` pre-tokenizer type:
//! this pair *is* what one lowers to.
//!
//! `read_metaspace`, in `tk-serialize`'s `from_json::pre_tokenizers`, builds that pair and spells
//! out which `Metaspace` settings can be rebuilt this way.
//!
//! [`Split`]: crate::pre_tokenizers::split::Split

use std::borrow::Cow;

use crate::tokenizer::{Result, pipeline};

/// Whether a `Metaspace` writes the delimiter at the start of a text that has no leading space.
/// [`First`](Self::First) writes it only at the start of the sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrependScheme {
    Always,
    First,
    Never,
}

/// Writes the delimiter where words start (after a space)
#[derive(Debug, Clone, PartialEq)]
pub struct MetaspaceNormalizer {
    /// `▁` (U+2581) for every SentencePiece model we know of, but the config is free to use
    /// another character.
    delimiter: char,
    /// Write the delimiter at the start of every word, not only the words that followed a space.
    prepend: PrependScheme,
    /// Throw whitespace away instead of turning it into a delimiter: tabs, newlines and repeated
    /// spaces leave no trace, and each word keeps only the one delimiter `prepend` writes. This is
    /// the [`WhitespaceSplit`] that t5 and albert run in front of their `Metaspace`.
    drop_whitespace: bool,
}

impl MetaspaceNormalizer {
    pub fn new(delimiter: char, prepend: PrependScheme, drop_whitespace: bool) -> Self {
        Self {
            delimiter,
            prepend,
            drop_whitespace,
        }
    }
}

impl MetaspaceNormalizer {
    /// The delimiter this writes, `\u{2581}` for every SentencePiece model we know of.
    pub fn delimiter(&self) -> char {
        self.delimiter
    }

    /// Whether the delimiter goes at the start of every word rather than only after a space.
    pub fn prepend(&self) -> PrependScheme {
        self.prepend
    }

    /// Whether whitespace is thrown away instead of becoming a delimiter, i.e. whether a
    /// `WhitespaceSplit` ran in front of the `Metaspace` this came from.
    pub fn drop_whitespace(&self) -> bool {
        self.drop_whitespace
    }
}

impl pipeline::Normalizer for MetaspaceNormalizer {
    fn normalize<'a>(&self, input: &'a str, offset: usize) -> Result<Cow<'a, str>> {
        // Return empty input as is
        if input.is_empty() {
            return Ok(Cow::Borrowed(input));
        }
        // The delimiter is 3 bytes where a space is 1, so the rewrite grows by 2 bytes per space, hence we allocate a bit more space
        let mut rewritten = String::with_capacity(input.len() + input.len() / 2);
        if self.drop_whitespace {
            let mut opens_the_sequence = offset == 0 && !input.starts_with(char::is_whitespace);
            for word in input.split_whitespace() {
                let prepend = match self.prepend {
                    PrependScheme::Always => true,
                    PrependScheme::First => opens_the_sequence,
                    PrependScheme::Never => false,
                };
                if prepend && !word.starts_with(self.delimiter) {
                    rewritten.push(self.delimiter);
                }
                opens_the_sequence = false;
                rewritten.push_str(word);
            }
        } else {
            let prepend = match self.prepend {
                PrependScheme::Always => true,
                PrependScheme::First => offset == 0,
                PrependScheme::Never => false,
            };
            if prepend && !input.starts_with(' ') && !input.starts_with(self.delimiter) {
                rewritten.push(self.delimiter);
            }
            // Only spaces become delimiters; tabs and newlines are left alone
            let mut rest = input;
            while let Some(space) = memchr::memchr(b' ', rest.as_bytes()) {
                rewritten.push_str(&rest[..space]);
                rewritten.push(self.delimiter);
                rest = &rest[space + 1..];
            }
            rewritten.push_str(rest);
        }
        Ok(Cow::Owned(rewritten))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::pipeline::Normalizer as _;

    /// `(prepend, drop_whitespace, offset, input, output)`, every expectation read off released
    /// `tokenizers` 0.23.1. `offset` is where `input` starts in its sequence.
    #[rustfmt::skip]
    const CASES: &[(PrependScheme, bool, usize, &str, &str)] = &[
        (PrependScheme::Always, false, 0, "aa bb cc",   "▁aa▁bb▁cc"),
        (PrependScheme::Always, false, 0, " aa bb",     "▁aa▁bb"),
        (PrependScheme::Always, false, 0, "aa\tbb  cc", "▁aa\tbb▁▁cc"),
        (PrependScheme::Always, false, 7, "aa bb cc",   "▁aa▁bb▁cc"),
        (PrependScheme::Never,  false, 0, "aa bb cc",   "aa▁bb▁cc"),
        (PrependScheme::Never,  false, 0, " aa bb",     "▁aa▁bb"),
        (PrependScheme::First,  false, 0, "aa bb cc",   "▁aa▁bb▁cc"),
        (PrependScheme::First,  false, 7, "aa bb cc",   "aa▁bb▁cc"),
        (PrependScheme::First,  false, 0, " aa bb",     "▁aa▁bb"),

        (PrependScheme::Always, true,  0, "aa bb cc",   "▁aa▁bb▁cc"),
        (PrependScheme::Always, true,  0, "aa\tbb  cc", "▁aa▁bb▁cc"),
        (PrependScheme::Never,  true,  0, "aa bb cc",   "aabbcc"),
        // A sequence opening with whitespace gets none: `WhitespaceSplit` removes it, moving the
        // first word off byte zero.
        (PrependScheme::First,  true,  0, "aa bb cc",   "▁aabbcc"),
        (PrependScheme::First,  true,  7, "aa bb cc",   "aabbcc"),
        (PrependScheme::First,  true,  0, " aa bb",     "aabb"),
    ];

    #[test]
    fn the_delimiter_lands_where_the_released_crate_puts_it() {
        for (prepend, drop_whitespace, offset, input, expected) in CASES {
            let normalizer = MetaspaceNormalizer::new('▁', *prepend, *drop_whitespace);
            assert_eq!(
                normalizer.normalize(input, *offset).unwrap(),
                *expected,
                "{prepend:?} drop_whitespace={drop_whitespace} offset={offset} {input:?}"
            );
        }
    }
}
