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

/// Whether a `Metaspace` writes the delimiter at the start of a text that has no leading space.
/// [`First`](Self::First) writes it only at the start of the sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PrependScheme {
    Always,
    First,
    Never,
}

/// Writes the delimiter where words start (after a space)
#[derive(Debug, Clone, PartialEq)]
pub struct Metaspace {
    /// `▁` (U+2581) for every SentencePiece model we know of, but the config is free to use
    /// another character.
    delimiter: char,
    /// Write the delimiter at the start of every word, not only the words that followed a space.
    prepend: PrependScheme,
    /// Throw whitespace away instead of turning it into a delimiter: tabs, newlines and repeated
    /// spaces leave no trace, and each word keeps only the one delimiter `prepend` writes. This is
    /// the [`WhitespaceSplit`] that t5 and albert run in front of their `Metaspace`.
    drop_whitespace: bool,
    /// Write the delimiter in front of a piece even when the piece already starts with one, which
    /// is what SentencePiece's `add_dummy_prefix` does. Without it a piece that opens with the
    /// delimiter keeps the one it has, like released `Metaspace` did.
    add_dummy_prefix: bool,
    /// Write the delimiter where the spaces were. SentencePiece's `escape_whitespaces: false`
    /// leaves the spaces alone, and writes a plain space as the dummy prefix.
    escape_whitespaces: bool,
    /// Strip the spaces that open and close a piece and write one delimiter for a run of them.
    /// Only spaces count, so tabs and newlines are left alone: SentencePiece's
    /// `remove_extra_whitespaces`. Ignored when `drop_whitespace` already throws everything away.
    remove_extra_whitespaces: bool,
}

impl Metaspace {
    pub fn new(delimiter: char, prepend: PrependScheme, drop_whitespace: bool) -> Self {
        Self {
            delimiter,
            prepend,
            drop_whitespace,
            add_dummy_prefix: false,
            escape_whitespaces: true,
            remove_extra_whitespaces: false,
        }
    }

    pub fn with_add_dummy_prefix(mut self, add_dummy_prefix: bool) -> Self {
        self.add_dummy_prefix = add_dummy_prefix;
        self
    }

    pub fn with_escape_whitespaces(mut self, escape_whitespaces: bool) -> Self {
        self.escape_whitespaces = escape_whitespaces;
        self
    }

    pub fn with_remove_extra_whitespaces(mut self, remove_extra_whitespaces: bool) -> Self {
        self.remove_extra_whitespaces = remove_extra_whitespaces;
        self
    }
}

impl Metaspace {
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

    /// Whether the delimiter goes in front of a piece that already starts with one.
    pub fn add_dummy_prefix(&self) -> bool {
        self.add_dummy_prefix
    }

    /// Whether spaces become the delimiter. When false they stay spaces, and so does the prefix.
    pub fn escape_whitespaces(&self) -> bool {
        self.escape_whitespaces
    }

    /// Whether the spaces around a piece are stripped and a run of them becomes one delimiter.
    pub fn remove_extra_whitespaces(&self) -> bool {
        self.remove_extra_whitespaces
    }
}

impl Metaspace {
    /// Whether the delimiter goes in front of `piece`. `add_dummy_prefix` makes it unconditional,
    /// which is what SentencePiece does; otherwise a piece that already opens with a space or with
    /// the delimiter keeps what it has.
    fn writes_prefix(&self, piece: &str, opens_sequence: bool, delimiter: char) -> bool {
        let prepend = match self.prepend {
            PrependScheme::Always => true,
            PrependScheme::First => opens_sequence,
            PrependScheme::Never => false,
        };
        prepend
            && (self.add_dummy_prefix || !(piece.starts_with(' ') || piece.starts_with(delimiter)))
    }

    /// Appends the rewritten `input` to `rewritten`. `opens_sequence` says whether `input` starts
    /// its sequence, which is what [`PrependScheme::First`] acts on.
    pub fn normalize_into(&self, input: &str, opens_sequence: bool, rewritten: &mut String) {
        if input.is_empty() {
            return;
        }
        let delimiter = if self.escape_whitespaces {
            self.delimiter
        } else {
            ' '
        };
        let written = rewritten.len();
        if self.drop_whitespace {
            let mut opens_the_sequence = opens_sequence && !input.starts_with(char::is_whitespace);
            for word in input.split_whitespace() {
                if self.writes_prefix(word, opens_the_sequence, delimiter) {
                    rewritten.push(delimiter);
                }
                opens_the_sequence = false;
                rewritten.push_str(word);
            }
        } else {
            // SentencePiece strips the spaces that open a piece before it writes the dummy prefix
            let rest = if self.remove_extra_whitespaces {
                input.trim_start_matches(' ')
            } else {
                input
            };
            if rest.is_empty() {
                return;
            }
            if self.writes_prefix(rest, opens_sequence, delimiter) {
                rewritten.push(delimiter);
            }
            if self.remove_extra_whitespaces {
                // A run of spaces becomes one delimiter. Only spaces count: tabs and newlines are
                // left alone, as in SentencePiece.
                let mut previous_was_space = true;
                for ch in rest.chars() {
                    if ch == ' ' {
                        if !previous_was_space {
                            rewritten.push(delimiter);
                        }
                        previous_was_space = true;
                    } else {
                        rewritten.push(ch);
                        previous_was_space = false;
                    }
                }
                // SentencePiece strips what closes the text last, so a delimiter the input itself
                // carried goes too. `written` keeps that off the piece before this one.
                while rewritten.len() > written && rewritten.ends_with(delimiter) {
                    rewritten.pop();
                }
            } else if self.escape_whitespaces {
                // Only spaces become delimiters; tabs and newlines are left alone
                let mut rest = rest;
                while let Some(space) = memchr::memchr(b' ', rest.as_bytes()) {
                    rewritten.push_str(&rest[..space]);
                    rewritten.push(delimiter);
                    rest = &rest[space + 1..];
                }
                rewritten.push_str(rest);
            } else {
                rewritten.push_str(rest);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(prepend, drop_whitespace, opens_sequence, input, output)`, every expectation read off
    /// released `tokenizers` 0.23.1.
    #[rustfmt::skip]
    const CASES: &[(PrependScheme, bool, bool, &str, &str)] = &[
        (PrependScheme::Always, false, true,  "aa bb cc",   "▁aa▁bb▁cc"),
        (PrependScheme::Always, false, true,  " aa bb",     "▁aa▁bb"),
        (PrependScheme::Always, false, true,  "aa\tbb  cc", "▁aa\tbb▁▁cc"),
        (PrependScheme::Always, false, false, "aa bb cc",   "▁aa▁bb▁cc"),
        (PrependScheme::Never,  false, true,  "aa bb cc",   "aa▁bb▁cc"),
        (PrependScheme::Never,  false, true,  " aa bb",     "▁aa▁bb"),
        (PrependScheme::First,  false, true,  "aa bb cc",   "▁aa▁bb▁cc"),
        (PrependScheme::First,  false, false, "aa bb cc",   "aa▁bb▁cc"),
        (PrependScheme::First,  false, true,  " aa bb",     "▁aa▁bb"),

        (PrependScheme::Always, true,  true,  "aa bb cc",   "▁aa▁bb▁cc"),
        (PrependScheme::Always, true,  true,  "aa\tbb  cc", "▁aa▁bb▁cc"),
        (PrependScheme::Never,  true,  true,  "aa bb cc",   "aabbcc"),
        // A sequence opening with whitespace gets none: `WhitespaceSplit` removes it, moving the
        // first word off byte zero.
        (PrependScheme::First,  true,  true,  "aa bb cc",   "▁aabbcc"),
        (PrependScheme::First,  true,  false, "aa bb cc",   "aabbcc"),
        (PrependScheme::First,  true,  true,  " aa bb",     "aabb"),
    ];

    #[test]
    fn the_delimiter_lands_where_the_released_crate_puts_it() {
        for (prepend, drop_whitespace, opens_sequence, input, expected) in CASES {
            let normalizer = Metaspace::new('▁', *prepend, *drop_whitespace);
            let mut out = String::new();
            normalizer.normalize_into(input, *opens_sequence, &mut out);
            assert_eq!(
                out, *expected,
                "{prepend:?} drop_whitespace={drop_whitespace} opens_sequence={opens_sequence} {input:?}"
            );
        }
    }

    /// `(prepend, add_dummy_prefix, escape_whitespaces, remove_extra_whitespaces, opens_sequence,
    /// input, output)`, every expectation read off SentencePiece 0.2.1
    /// `SentencePieceProcessor.normalize` on a model trained with `--normalization_rule_name=identity`
    /// and those flags. SentencePiece's `add_dummy_prefix: false` is `prepend: "never"` here, and
    /// `add_dummy_prefix: true` is `prepend: "always"` plus the flag that makes it unconditional.
    #[rustfmt::skip]
    const SENTENCEPIECE_CASES: &[(PrependScheme, bool, bool, bool, bool, &str, &str)] = &[
        (PrependScheme::Always, true, true, true, true,  "aa bb cc",   "▁aa▁bb▁cc"),
        (PrependScheme::Always, true, true, true, true,  " aa bb",     "▁aa▁bb"),
        (PrependScheme::Always, true, true, true, true,  "aa\tbb  cc", "▁aa\tbb▁cc"),
        (PrependScheme::Always, true, true, true, true,  "aa\nbb",     "▁aa\nbb"),
        // A dummy prefix goes in front of a text that already opens with the delimiter.
        (PrependScheme::Always, true, true, true, true,  "▁aa",        "▁▁aa"),
        // ... and the trailing strip takes the delimiter the text itself carried.
        (PrependScheme::Always, true, true, true, true,  "a▁",         "▁a"),
        (PrependScheme::Always, true, true, true, true,  "   aa   ",   "▁aa"),
        (PrependScheme::Always, true, true, true, true,  "  ",         ""),
        (PrependScheme::Always, true, true, true, true,  "",           ""),
        (PrependScheme::Always, true, true, true, true,  "aa",         "▁aa"),

        (PrependScheme::Always, true, true, false, true, "aa bb cc",   "▁aa▁bb▁cc"),
        (PrependScheme::Always, true, true, false, true, " aa bb",     "▁▁aa▁bb"),
        (PrependScheme::Always, true, true, false, true, "aa\tbb  cc", "▁aa\tbb▁▁cc"),
        (PrependScheme::Always, true, true, false, true, "aa\nbb",     "▁aa\nbb"),
        (PrependScheme::Always, true, true, false, true, "▁aa",        "▁▁aa"),
        (PrependScheme::Always, true, true, false, true, "a▁",         "▁a▁"),
        (PrependScheme::Always, true, true, false, true, "  ",         "▁▁▁"),
        (PrependScheme::Always, true, true, false, true, "   aa   ",   "▁▁▁▁aa▁▁▁"),
        (PrependScheme::Always, true, true, false, true, "",           ""),

        (PrependScheme::Never,  false, true, true, true, "aa bb cc",   "aa▁bb▁cc"),
        (PrependScheme::Never,  false, true, true, true, " aa bb",     "aa▁bb"),
        (PrependScheme::Never,  false, true, true, true, "aa\tbb  cc", "aa\tbb▁cc"),
        (PrependScheme::Never,  false, true, true, true, "▁aa",        "▁aa"),
        (PrependScheme::Never,  false, true, true, true, "a▁",         "a"),
        (PrependScheme::Never,  false, true, true, true, "   aa   ",   "aa"),
        (PrependScheme::Never,  false, true, true, true, "  ",         ""),

        // Not from SentencePiece: it refuses to train with `escape_whitespaces` off. Read
        // literally, the flag leaves the spaces alone and writes a space as the dummy prefix.
        (PrependScheme::Always, true, false, true, true,  "aa bb",      " aa bb"),
        (PrependScheme::Never,  false, false, false, true, "aa bb",     "aa bb"),

        // `first` is the non-legacy chunking of transformers: only the piece that opens the
        // sequence gets the dummy prefix.
        (PrependScheme::First,  true, true, true, true,  "aa bb",      "▁aa▁bb"),
        (PrependScheme::First,  true, true, true, false, "aa bb",      "aa▁bb"),
    ];

    #[test]
    fn the_piece_matches_sentencepieces_normalizer() {
        for (prepend, dummy, escape, remove_extra, opens_sequence, input, expected) in
            SENTENCEPIECE_CASES
        {
            let normalizer = Metaspace::new('▁', *prepend, false)
                .with_add_dummy_prefix(*dummy)
                .with_escape_whitespaces(*escape)
                .with_remove_extra_whitespaces(*remove_extra);
            let mut out = String::new();
            normalizer.normalize_into(input, *opens_sequence, &mut out);
            assert_eq!(
                out, *expected,
                "{prepend:?} add_dummy_prefix={dummy} escape_whitespaces={escape} \
                 remove_extra_whitespaces={remove_extra} opens_sequence={opens_sequence} {input:?}"
            );
        }
    }
}
