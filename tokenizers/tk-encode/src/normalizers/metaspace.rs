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

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum PrependBehavior {
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
    prepend: PrependBehavior,
    /// Throw whitespace away instead of turning it into a delimiter: tabs, newlines and repeated
    /// spaces leave no trace, and each word keeps only the one delimiter `prepend` writes. This is
    /// the [`WhitespaceSplit`] that t5 and albert run in front of their `Metaspace`.
    drop_whitespace: bool,
}

impl MetaspaceNormalizer {
    pub fn new(delimiter: char, prepend: PrependBehavior, drop_whitespace: bool) -> Self {
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
    pub fn prepend(&self) -> PrependBehavior {
        self.prepend
    }

    /// Whether whitespace is thrown away instead of becoming a delimiter, i.e. whether a
    /// `WhitespaceSplit` ran in front of the `Metaspace` this came from.
    pub fn drop_whitespace(&self) -> bool {
        self.drop_whitespace
    }
}

impl pipeline::Normalizer for MetaspaceNormalizer {
    fn normalize<'a>(&self, input: &'a str, is_sequence_start: bool) -> Result<Cow<'a, str>> {
        // Return empty input as is
        if input.is_empty() {
            return Ok(Cow::Borrowed(input));
        }
        // The delimiter is 3 bytes where a space is 1, so the rewrite grows by 2 bytes per space, hence we allocate a bit more space
        let mut rewritten = String::with_capacity(input.len() + input.len() / 2);
        if self.drop_whitespace {
            let mut opens_the_sequence =
                is_sequence_start && !input.starts_with(char::is_whitespace);
            for word in input.split_whitespace() {
                let prepend = match self.prepend {
                    PrependBehavior::Always => true,
                    PrependBehavior::First => opens_the_sequence,
                    PrependBehavior::Never => false,
                };
                if prepend && !word.starts_with(self.delimiter) {
                    rewritten.push(self.delimiter);
                }
                opens_the_sequence = false;
                rewritten.push_str(word);
            }
        } else {
            let prepend = match self.prepend {
                PrependBehavior::Always => true,
                PrependBehavior::First => is_sequence_start,
                PrependBehavior::Never => false,
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

    /// `(prepend, drop_whitespace, is_sequence_start, input, output)`.
    ///
    /// Every expectation is where the released `tokenizers` 0.23.1 puts the delimiter, read off a
    /// `Metaspace` pre-tokenizer (and a `Sequence[WhitespaceSplit, Metaspace]` for the
    /// `drop_whitespace` rows) and written out as the one string this normalizer produces.
    #[rustfmt::skip]
    const CASES: &[(PrependBehavior, bool, bool, &str, &str)] = &[
        (PrependBehavior::Always, false, true,  "aa bb cc",   "▁aa▁bb▁cc"),
        (PrependBehavior::Always, false, true,  " aa bb",     "▁aa▁bb"),
        (PrependBehavior::Always, false, true,  "aa\tbb  cc", "▁aa\tbb▁▁cc"),
        (PrependBehavior::Never,  false, true,  "aa bb cc",   "aa▁bb▁cc"),
        (PrependBehavior::Never,  false, true,  " aa bb",     "▁aa▁bb"),
        // `First` at the start of the sequence is `Always` for the first word and `Never` after it.
        (PrependBehavior::First,  false, true,  "aa bb cc",   "▁aa▁bb▁cc"),
        (PrependBehavior::First,  false, false, "aa bb cc",   "aa▁bb▁cc"),
        (PrependBehavior::First,  false, true,  " aa bb",     "▁aa▁bb"),

        (PrependBehavior::Always, true,  true,  "aa bb cc",   "▁aa▁bb▁cc"),
        (PrependBehavior::Always, true,  true,  "aa\tbb  cc", "▁aa▁bb▁cc"),
        (PrependBehavior::Never,  true,  true,  "aa bb cc",   "aabbcc"),
        // Only the word that opens the sequence, and a sequence opening with whitespace has none:
        // `WhitespaceSplit` removes it, which moves the first word off byte zero.
        (PrependBehavior::First,  true,  true,  "aa bb cc",   "▁aabbcc"),
        (PrependBehavior::First,  true,  false, "aa bb cc",   "aabbcc"),
        (PrependBehavior::First,  true,  true,  " aa bb",     "aabb"),
    ];

    #[test]
    fn the_delimiter_lands_where_the_released_crate_puts_it() {
        for (prepend, drop_whitespace, is_sequence_start, input, expected) in CASES {
            let normalizer = MetaspaceNormalizer::new('▁', *prepend, *drop_whitespace);
            assert_eq!(
                normalizer.normalize(input, *is_sequence_start).unwrap(),
                *expected,
                "{prepend:?} drop_whitespace={drop_whitespace} \
                 is_sequence_start={is_sequence_start} {input:?}"
            );
        }
    }
}
