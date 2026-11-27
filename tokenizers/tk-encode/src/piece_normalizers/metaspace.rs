//! Writes SentencePiece's `▁` into the text, where the spaces used to be.
//!
//! SentencePiece tokenizers have no token for a space.
//! A space is folded into the word that follows it and written `▁`.
//! That character marks where a word starts *and* separates words, so the code below calls it the delimiter.
//!
//! In a version 1.0 `tokenizer.json`, one `Metaspace` pre-tokenizer rewrites the text and cuts it into words.
//! tk-convert rebuilds it as this piece normalizer plus a [`Split`] on the delimiter.
//! As a piece normalizer, it writes the delimiter on each piece of text between added tokens, never in front of an added token.
//!
//! `add_dummy_prefix`, `escape_whitespaces` and `remove_extra_whitespaces` are SentencePiece's own
//! normalizer flags, so that a tokenizer converted from a `.model` file needs no other normalizer
//! than the charsmap in front of this one: the whole of `Normalizer::Normalize` (minus the
//! charsmap) happens here, in one pass.
//!
//! [`Split`]: crate::pre_tokenizers::split::Split

use std::iter;

/// Whether a `Metaspace` writes the delimiter at the start of a piece that has no leading space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PrependScheme {
    /// Prefix every piece with the delimiter
    Always,
    /// Only prefix the piece that starts at index 0 in the input sequence
    First,
    /// Do not prefix pieces
    Never,
}

/// What is matched as a whitespace in the Metaspace piece normalizer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WhitespaceMatch {
    /// Only the ASCII space (`U+0020`) is matched as a whitespace
    OnlySpace,
    /// Any unicode whitespace is matched as a whitespace
    AnyUnicode,
}

/// Writes the delimiter where words start (after a space)
#[derive(Debug, Clone, PartialEq)]
pub struct Metaspace {
    /// `▁` (U+2581) for every SentencePiece model we know of, but the config is free to use
    /// another character.
    delimiter: char,
    /// Whether to prepend pieces with the delimiter
    prepend: PrependScheme,
    /// What is matched as a whitespace in the piece normalizer
    whitespace_match: WhitespaceMatch,
    /// Whether to collapse a consecutive run of whitespaces into a single space
    collapse_contiguous: bool,
    /// Whether to skip the prepend if the piece already starts with the delimiter
    skip_prefix: bool,
    /// Whether to drop the delimiters that end the piece, including ones the piece already had
    trim_trailing_delimiters: bool,
}

impl Metaspace {
    pub fn new(
        delimiter: char,
        prepend: PrependScheme,
        whitespace_match: WhitespaceMatch,
        collapse_contiguous: bool,
        skip_prefix: bool,
        trim_trailing_delimiters: bool,
    ) -> Self {
        Self {
            delimiter,
            prepend,
            whitespace_match,
            collapse_contiguous,
            skip_prefix,
            trim_trailing_delimiters,
        }
    }

    /// Builds the `Metaspace` that rewrites whitespace like SentencePiece's `Normalizer::Normalize`,
    /// from the flags in a `.model` file's `normalizer_spec`.
    ///
    /// With `escape_whitespaces` off, spaces stay spaces and the dummy prefix is a space.
    pub fn from_sentencepiece(
        add_dummy_prefix: bool,
        remove_extra_whitespaces: bool,
        escape_whitespaces: bool,
    ) -> Self {
        Self {
            delimiter: if escape_whitespaces { '▁' } else { ' ' },
            prepend: if add_dummy_prefix {
                PrependScheme::Always
            } else {
                PrependScheme::Never
            },
            whitespace_match: WhitespaceMatch::OnlySpace,
            collapse_contiguous: remove_extra_whitespaces,
            skip_prefix: false,
            trim_trailing_delimiters: remove_extra_whitespaces,
        }
    }
}

impl Metaspace {
    /// The delimiter this writes, `\u{2581}` for every SentencePiece model we know of.
    pub fn delimiter(&self) -> char {
        self.delimiter
    }

    /// Which pieces get the delimiter in front.
    pub fn prepend(&self) -> PrependScheme {
        self.prepend
    }

    /// Which characters are rewritten as the delimiter.
    pub fn whitespace_match(&self) -> WhitespaceMatch {
        self.whitespace_match
    }

    /// Whether a run of whitespace becomes one delimiter, with the runs at the piece's edges dropped.
    pub fn collapse_contiguous(&self) -> bool {
        self.collapse_contiguous
    }

    /// Whether the prefix is left out when the piece already starts with whitespace or the delimiter.
    pub fn skip_prefix(&self) -> bool {
        self.skip_prefix
    }

    /// Whether the delimiters that end the piece are dropped, including ones the piece already had.
    pub fn trim_trailing_delimiters(&self) -> bool {
        self.trim_trailing_delimiters
    }
}

impl Metaspace {
    fn is_whitespace(&self, ch: char) -> bool {
        match self.whitespace_match {
            WhitespaceMatch::OnlySpace => ch == ' ',
            WhitespaceMatch::AnyUnicode => ch.is_whitespace(),
        }
    }

    /// Whether `text` keeps the whitespace or delimiter it starts with, in place of a new delimiter.
    fn already_delimited(&self, text: &str) -> bool {
        self.skip_prefix && text.starts_with(|ch| self.is_whitespace(ch) || ch == self.delimiter)
    }

    /// Appends the rewritten `piece` to `out`. `opens_sequence` says whether `piece` starts its
    /// sequence, which is what [`PrependScheme::First`] acts on.
    pub fn normalize_into(&self, piece: &str, opens_sequence: bool, out: &mut String) {
        let add_prefix = match self.prepend {
            PrependScheme::Always => true,
            PrependScheme::First => opens_sequence,
            PrependScheme::Never => false,
        };
        let piece_start = out.len();
        match self.whitespace_match {
            WhitespaceMatch::OnlySpace => {
                self.replace_whitespace(piece, split_at_spaces(piece), add_prefix, out)
            }
            WhitespaceMatch::AnyUnicode => {
                self.replace_whitespace(piece, piece.split(char::is_whitespace), add_prefix, out)
            }
        }
        if self.trim_trailing_delimiters {
            while out.len() > piece_start && out.ends_with(self.delimiter) {
                out.pop();
            }
        }
    }

    /// Writes `piece` with the delimiter in place of each whitespace character.
    /// `parts` is the text between those whitespace characters.
    fn replace_whitespace<'a>(
        &self,
        piece: &str,
        parts: impl Iterator<Item = &'a str>,
        add_prefix: bool,
        out: &mut String,
    ) {
        if self.collapse_contiguous {
            let words = parts.filter(|part| !part.is_empty());
            for (index, word) in words.enumerate() {
                let after_whitespace = index > 0;
                let wants_delimiter = after_whitespace || add_prefix;
                if wants_delimiter && !self.already_delimited(word) {
                    out.push(self.delimiter);
                }
                out.push_str(word);
            }
        } else {
            if add_prefix && !piece.is_empty() && !self.already_delimited(piece) {
                out.push(self.delimiter);
            }
            for (index, part) in parts.enumerate() {
                let after_whitespace = index > 0;
                if after_whitespace {
                    out.push(self.delimiter);
                }
                out.push_str(part);
            }
        }
    }
}

/// The parts of `piece` between its spaces, as `piece.split(' ')` gives them.
// std's `split` finds the spaces without SIMD, which is slower on text with few of them.
fn split_at_spaces(piece: &str) -> impl Iterator<Item = &str> {
    let ends = memchr::memchr_iter(b' ', piece.as_bytes()).chain(iter::once(piece.len()));
    let mut start = 0;
    ends.map(move |end| {
        let part = &piece[start..end];
        start = end + 1;
        part
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(prepend, drop_whitespace, opens_sequence, input, output)`, every expectation read off
    /// released `tokenizers` 0.23.1. `drop_whitespace` is a `WhitespaceSplit` in front of the
    /// `Metaspace`, which the Hub only ships with `prepend: always`.
    #[rustfmt::skip]
    const CASES: &[(PrependScheme, bool, bool, &str, &str)] = &[
        (PrependScheme::Always, false, true,  "aa bb cc",      "▁aa▁bb▁cc"),
        (PrependScheme::Always, false, true,  " aa bb",        "▁aa▁bb"),
        (PrependScheme::Always, false, true,  "aa\tbb  cc",    "▁aa\tbb▁▁cc"),
        (PrependScheme::Always, false, false, "aa bb cc",      "▁aa▁bb▁cc"),
        (PrependScheme::Never,  false, true,  "aa bb cc",      "aa▁bb▁cc"),
        (PrependScheme::Never,  false, true,  " aa bb",        "▁aa▁bb"),
        (PrependScheme::First,  false, true,  "aa bb cc",      "▁aa▁bb▁cc"),
        (PrependScheme::First,  false, false, "aa bb cc",      "aa▁bb▁cc"),
        (PrependScheme::First,  false, true,  " aa bb",        "▁aa▁bb"),

        (PrependScheme::Always, true,  true,  "aa bb cc",      "▁aa▁bb▁cc"),
        (PrependScheme::Always, true,  true,  "aa\tbb  cc",    "▁aa▁bb▁cc"),
        // A word that opens with the delimiter gets no other, and a trailing one stays.
        (PrependScheme::Always, true,  true,  "hello ▁ world", "▁hello▁▁world"),
        (PrependScheme::Always, true,  true,  "hello▁",        "▁hello▁"),
    ];

    #[test]
    fn the_delimiter_lands_where_the_released_crate_puts_it() {
        for (prepend, drop_whitespace, opens_sequence, input, expected) in CASES {
            let whitespace_match = if *drop_whitespace {
                WhitespaceMatch::AnyUnicode
            } else {
                WhitespaceMatch::OnlySpace
            };
            let normalizer = Metaspace::new(
                '▁',
                *prepend,
                whitespace_match,
                *drop_whitespace,
                true,
                false,
            );
            let mut out = String::new();
            normalizer.normalize_into(input, *opens_sequence, &mut out);
            assert_eq!(
                out, *expected,
                "{prepend:?} drop_whitespace={drop_whitespace} opens_sequence={opens_sequence} {input:?}"
            );
        }
    }

    /// `(add_dummy_prefix, escape_whitespaces, remove_extra_whitespaces, input, output)`, every
    /// expectation read off SentencePiece 0.2.1 `SentencePieceProcessor.normalize` on a model
    /// trained with `--normalization_rule_name=identity` and those flags.
    #[rustfmt::skip]
    const SENTENCEPIECE_CASES: &[(bool, bool, bool, &str, &str)] = &[
        (true, true, true,  "aa bb cc",   "▁aa▁bb▁cc"),
        (true, true, true,  " aa bb",     "▁aa▁bb"),
        (true, true, true,  "aa\tbb  cc", "▁aa\tbb▁cc"),
        (true, true, true,  "aa\nbb",     "▁aa\nbb"),
        // A dummy prefix goes in front of a text that already opens with the delimiter.
        (true, true, true,  "▁aa",        "▁▁aa"),
        // ... and the trailing strip takes the delimiter the text itself carried.
        (true, true, true,  "a▁",         "▁a"),
        (true, true, true,  "   aa   ",   "▁aa"),
        (true, true, true,  "  ",         ""),
        (true, true, true,  "",           ""),
        (true, true, true,  "aa",         "▁aa"),

        (true, true, false, "aa bb cc",   "▁aa▁bb▁cc"),
        (true, true, false, " aa bb",     "▁▁aa▁bb"),
        (true, true, false, "aa\tbb  cc", "▁aa\tbb▁▁cc"),
        (true, true, false, "aa\nbb",     "▁aa\nbb"),
        (true, true, false, "▁aa",        "▁▁aa"),
        (true, true, false, "a▁",         "▁a▁"),
        (true, true, false, "  ",         "▁▁▁"),
        (true, true, false, "   aa   ",   "▁▁▁▁aa▁▁▁"),
        (true, true, false, "",           ""),

        (false, true, true, "aa bb cc",   "aa▁bb▁cc"),
        (false, true, true, " aa bb",     "aa▁bb"),
        (false, true, true, "aa\tbb  cc", "aa\tbb▁cc"),
        (false, true, true, "▁aa",        "▁aa"),
        (false, true, true, "a▁",         "a"),
        (false, true, true, "   aa   ",   "aa"),
        (false, true, true, "  ",         ""),

        // Not from SentencePiece: it refuses to train with `escape_whitespaces` off. Read
        // literally, the flag leaves the spaces alone and writes a space as the dummy prefix.
        (true,  false, true,  "aa bb",    " aa bb"),
        (false, false, false, "aa bb",    "aa bb"),
    ];

    #[test]
    fn the_piece_matches_sentencepieces_normalizer() {
        for (dummy, escape, remove_extra, input, expected) in SENTENCEPIECE_CASES {
            let normalizer = Metaspace::from_sentencepiece(*dummy, *remove_extra, *escape);
            let mut out = String::new();
            normalizer.normalize_into(input, true, &mut out);
            assert_eq!(
                out, *expected,
                "add_dummy_prefix={dummy} escape_whitespaces={escape} \
                 remove_extra_whitespaces={remove_extra} {input:?}"
            );
        }
    }
}
