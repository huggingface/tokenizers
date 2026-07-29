//! Writing SentencePiece's `▁` delimiters into the text.
//!
//! Built through [`crate::pre_tokenizers::metaspace::to_normalizer_and_split`]

use std::borrow::Cow;

use crate::tokenizer::{Result, pipeline};

#[derive(Debug, Clone, PartialEq)]
pub struct MetaspaceNormalizer {
    /// The delimiter for words start. Usually `▁` unless the tokenizer specifies another one.
    delimiter: char,
    /// Whether to prepend the delimiter to every word.
    /// `false` leaves the text as it is apart from the spaces.
    prepend: bool,
    /// todo: document
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
        if input.is_empty() {
            return Ok(Cow::Borrowed(input));
        }
        let mut rewritten = String::with_capacity(input.len() + input.len() / 4);
        // No double delimiter at the start of the word
        let open_word = |word: &str, rewritten: &mut String| {
            if self.prepend && !word.starts_with(self.delimiter) {
                rewritten.push(self.delimiter);
            }
            rewritten.push_str(word);
        };
        if self.drop_whitespace {
            let mut words = Vec::new();
            pipeline::classify_into_spans(
                input.as_bytes(),
                atomsplit::fsm::class_runs_into::<{ atomsplit::classify::mask::WS }, 0, 0>,
                &mut words,
            );
            for word in &words {
                open_word(&input[word.range()], &mut rewritten);
            }
        } else {
            if self.prepend && !input.starts_with(' ') && !input.starts_with(self.delimiter) {
                rewritten.push(self.delimiter);
            }
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
