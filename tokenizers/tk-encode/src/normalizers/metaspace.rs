//! Writes SentencePiece's `▁` into the text, where the spaces used to be.
//!
//! SentencePiece tokenizers have no token for a space: a space is folded into the word that follows
//! it and written `▁`. That character therefore marks where a word starts *and* separates words —
//! which is why the code below calls it the delimiter. In a tokenizer config, one [`Metaspace`]
//! pre-tokenizer does the whole job: it rewrites the text and cuts it into words. The pipeline keeps
//! those two apart — normalizers rewrite text, pre-tokenizers cut it — so a [`Metaspace`] is rebuilt
//! as this normalizer plus a [`Split`] on the delimiter.
//!
//! `to_normalizer_and_split`, in [`crate::pre_tokenizers::metaspace`], builds that pair and spells
//! out which [`Metaspace`] settings can be rebuilt this way.
//!
//! [`Metaspace`]: crate::pre_tokenizers::metaspace::Metaspace
//! [`Split`]: crate::pre_tokenizers::split::Split

use std::borrow::Cow;

use crate::pre_tokenizers::whitespace::WhitespaceSplit;
use crate::tokenizer::{Result, pipeline};

/// Writes the delimiter where words start (after a space)
#[derive(Debug, Clone, PartialEq)]
pub struct MetaspaceNormalizer {
    /// `▁` (U+2581) for every SentencePiece model we know of, but the config is free to use
    /// another character.
    delimiter: char,
    /// Write the delimiter at the start of every word, not only the words that followed a space.
    prepend: bool,
    /// Throw whitespace away instead of turning it into a delimiter: tabs, newlines and repeated
    /// spaces leave no trace, and each word keeps only the one delimiter `prepend` writes. This is
    /// the [`WhitespaceSplit`] that t5 and albert run in front of their `Metaspace`.
    drop_whitespace: bool,
}

impl MetaspaceNormalizer {
    pub(crate) fn new(delimiter: char, prepend: bool, drop_whitespace: bool) -> Self {
        Self {
            delimiter,
            prepend,
            drop_whitespace,
        }
    }
}

impl pipeline::Normalizer for MetaspaceNormalizer {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
        // Return empty input as is
        if input.is_empty() {
            return Ok(Cow::Borrowed(input));
        }
        // The delimiter is 3 bytes where a space is 1, so the rewrite grows by 2 bytes per space, hence we allocate a bit more space
        let mut rewritten = String::with_capacity(input.len() + input.len() / 2);
        if self.drop_whitespace {
            // Whitespace is thrown away, so cut the text where `WhitespaceSplit` would and write the
            // words back one after the other, each with its own delimiter.
            let mut words = Vec::new();
            pipeline::PreTokenizer::pre_tokenize(&WhitespaceSplit, input, &mut words)?;
            for span in &words {
                let word = &input[span.range()];
                // The text may already hold delimiters of its own — never write a second one.
                if self.prepend && !word.starts_with(self.delimiter) {
                    rewritten.push(self.delimiter);
                }
                rewritten.push_str(word);
            }
        } else {
            // Prepend the delimiter if self.prepend is true
            if self.prepend && !input.starts_with(' ') && !input.starts_with(self.delimiter) {
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
