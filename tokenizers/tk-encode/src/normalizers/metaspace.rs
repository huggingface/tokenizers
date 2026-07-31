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
//! Other configs spell the same rewrite in their `normalizer` field instead: llama-2 as
//! `Prepend("▁")` followed by `Replace(" " -> "▁")`, gemma-4 as the `Replace` alone. Run as
//! declared, every one of those steps writes a full copy of the text. `MetaspaceNormalizer::fuse`
//! recognizes both shapes when the pipeline is built and stands in for them: one pass, one
//! right-sized allocation.
//!
//! [`Metaspace`]: crate::pre_tokenizers::metaspace::Metaspace
//! [`Split`]: crate::pre_tokenizers::split::Split

use std::borrow::Cow;

use atomsplit::literal::Literal;

use crate::normalizers::NormalizerWrapper;
use crate::normalizers::replace::{Replace, ReplacePattern};
use crate::pre_tokenizers::whitespace::WhitespaceSplit;
use crate::tokenizer::{Result, pipeline};

/// When [`MetaspaceNormalizer`] writes a delimiter at the start of the text it is given.
///
/// The two prepending modes differ only on text that already starts with the delimiter (or with a
/// space the swap turns into one): `IfMissing` skips those, which is how a [`Metaspace`]
/// pre-tokenizer prepends; `Unconditional` marks them again, which is what a `Prepend` normalizer
/// running before the swap does.
///
/// [`Metaspace`]: crate::pre_tokenizers::metaspace::Metaspace
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum PrependMode {
    Never,
    IfMissing,
    Unconditional,
}

/// Writes the delimiter where words start (after a space)
#[derive(Debug, Clone, PartialEq)]
pub struct MetaspaceNormalizer {
    /// `▁` (U+2581) for every SentencePiece model we know of, but the config is free to use
    /// another character.
    delimiter: char,
    /// When to write the delimiter at the start of the text.
    prepend: PrependMode,
    /// Throw whitespace away instead of turning it into a delimiter: tabs, newlines and repeated
    /// spaces leave no trace, and each word keeps only the one delimiter `prepend` writes. This is
    /// the [`WhitespaceSplit`] that t5 and albert run in front of their `Metaspace`.
    drop_whitespace: bool,
}

impl MetaspaceNormalizer {
    pub(crate) fn new(delimiter: char, prepend: PrependMode, drop_whitespace: bool) -> Self {
        Self {
            delimiter,
            prepend,
            drop_whitespace,
        }
    }

    /// The one-pass stand-in for the leading `steps`, when they spell out this normalizer's job:
    /// `Prepend(c)` followed by `Replace(" " -> c)` (llama-2's normalizer), or a `Replace(" " -> c)`
    /// on its own (gemma-4's). Returns the stand-in and how many steps it covers; `None` when the
    /// steps start with anything else, and the caller keeps them as declared.
    ///
    /// [`PipelineTokenizer::try_from`] runs the config's normalizer steps through this when the
    /// pipeline is built. The config itself is never rewritten.
    ///
    /// [`PipelineTokenizer::try_from`]: crate::pipeline::PipelineTokenizer
    pub(crate) fn fuse(steps: &[NormalizerWrapper]) -> Option<(Self, usize)> {
        if let [
            NormalizerWrapper::Prepend(prepend),
            NormalizerWrapper::Replace(replace),
            ..,
        ] = steps
            && let Some(delimiter) = space_swap(replace)
            && prepend.prepend == replace.content
        {
            return Some((Self::new(delimiter, PrependMode::Unconditional, false), 2));
        }
        if let [NormalizerWrapper::Replace(replace), ..] = steps
            && let Some(delimiter) = space_swap(replace)
        {
            return Some((Self::new(delimiter, PrependMode::Never, false), 1));
        }
        None
    }
}

/// The delimiter a [`Replace`] swaps every space for: its pattern must be the string `" "` and its
/// content a single char. Anything else (a regex pattern, even one only matching a space; content
/// of any other length) is not checkable as the space swap and returns `None`.
fn space_swap(replace: &Replace) -> Option<char> {
    match &replace.pattern {
        ReplacePattern::String(pattern) if pattern == " " => {}
        _ => return None,
    }
    let mut chars = replace.content.chars();
    let delimiter = chars.next()?;
    chars.next().is_none().then_some(delimiter)
}

impl pipeline::Normalizer for MetaspaceNormalizer {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
        // Return empty input as is
        if input.is_empty() {
            return Ok(Cow::Borrowed(input));
        }
        if self.drop_whitespace {
            // Whitespace is thrown away, so cut the text where `WhitespaceSplit` would and write the
            // words back one after the other, each with its own delimiter.
            let mut words = Vec::new();
            pipeline::PreTokenizer::pre_tokenize(&WhitespaceSplit, input, &mut words)?;
            // Exact when every word takes a delimiter; a word that already starts with one
            // (`IfMissing`) leaves a few bytes spare.
            let words_len: usize = words.iter().map(|span| span.range().len()).sum();
            let mut rewritten =
                String::with_capacity(words_len + words.len() * self.delimiter.len_utf8());
            for span in &words {
                let word = &input[span.range()];
                let prepend = match self.prepend {
                    PrependMode::Never => false,
                    // The text may already hold delimiters of its own: never write a second one.
                    PrependMode::IfMissing => !word.starts_with(self.delimiter),
                    PrependMode::Unconditional => true,
                };
                if prepend {
                    rewritten.push(self.delimiter);
                }
                rewritten.push_str(word);
            }
            Ok(Cow::Owned(rewritten))
        } else {
            let prepend = match self.prepend {
                PrependMode::Never => false,
                // A leading space counts as already marked: the swap turns it into a delimiter.
                PrependMode::IfMissing => {
                    !input.starts_with(' ') && !input.starts_with(self.delimiter)
                }
                PrependMode::Unconditional => true,
            };
            // Only spaces become delimiters; tabs and newlines are left alone. Counting them
            // first sizes the rewrite exactly (a space is one byte, the delimiter up to four)
            // and lets the swap stream through the batch scan instead of restarting a search
            // at every space.
            let space = Literal::new(b" ").expect("a space is not empty");
            let count = space.count_matches(input.as_bytes());
            // Nothing to prepend and nothing to swap: hand the input back instead of copying it.
            if !prepend && count == 0 {
                return Ok(Cow::Borrowed(input));
            }
            let mut buf = [0u8; 4];
            let delimiter = self.delimiter.encode_utf8(&mut buf);
            let mut rewritten = String::with_capacity(
                input.len()
                    + (delimiter.len() - 1) * count
                    + if prepend { delimiter.len() } else { 0 },
            );
            if prepend {
                rewritten.push_str(delimiter);
            }
            let mut prev = 0;
            space.for_each_match(input.as_bytes(), |start| {
                rewritten.push_str(&input[prev..start]);
                rewritten.push_str(delimiter);
                prev = start + 1;
            });
            rewritten.push_str(&input[prev..]);
            Ok(Cow::Owned(rewritten))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::pipeline::normalize_all;

    fn step(json: &str) -> NormalizerWrapper {
        serde_json::from_str(json).unwrap()
    }

    /// llama-2's two normalizer steps.
    const PREPEND: &str = r#"{"type":"Prepend","prepend":"▁"}"#;
    /// gemma-4's whole normalizer, and llama-2's second step.
    const SWAP: &str = r#"{"type":"Replace","pattern":{"String":" "},"content":"▁"}"#;

    /// Every chunk shape the rewrite can meet: empty, already marked, leading, trailing and
    /// repeated spaces, no spaces at all, other whitespace.
    const TEXTS: &[&str] = &[
        "",
        "hello world",
        "hello   world",
        " leading",
        "trailing ",
        "  both  ",
        "▁already marked",
        "▁",
        "no_spaces",
        "one\ttab\nand a newline",
        "   ",
        "a▁b c",
    ];

    /// The stand-in must rewrite every text exactly as the steps it covers would.
    fn assert_fused_matches_steps(jsons: &[&str]) {
        let steps: Vec<NormalizerWrapper> = jsons.iter().map(|json| step(json)).collect();
        let (fused, covered) = MetaspaceNormalizer::fuse(&steps).expect("this shape fuses");
        assert_eq!(covered, steps.len());
        for text in TEXTS {
            let expected = normalize_all(&steps, text).unwrap();
            let got = pipeline::Normalizer::normalize(&fused, text).unwrap();
            assert_eq!(got, expected, "{text:?}");
        }
    }

    #[test]
    fn fused_prepend_and_swap_match_the_two_steps() {
        assert_fused_matches_steps(&[PREPEND, SWAP]);
    }

    #[test]
    fn fused_swap_matches_the_lone_step() {
        assert_fused_matches_steps(&[SWAP]);
    }

    /// Nothing ties the delimiter to `▁`; any single char must fuse the same way.
    #[test]
    fn fused_ascii_delimiter_matches_its_steps() {
        assert_fused_matches_steps(&[
            r#"{"type":"Prepend","prepend":"_"}"#,
            r#"{"type":"Replace","pattern":{"String":" "},"content":"_"}"#,
        ]);
    }

    /// The pair stands in for a `Prepend`, which marks a chunk even when it already starts with
    /// the delimiter; the lone swap never prepends. `covered` tells the caller how many steps to
    /// skip.
    #[test]
    fn the_pair_prepends_unconditionally_and_the_lone_swap_never_does() {
        let steps = [step(PREPEND), step(SWAP)];
        let (fused, covered) = MetaspaceNormalizer::fuse(&steps).unwrap();
        assert_eq!(covered, 2);
        assert_eq!(
            fused,
            MetaspaceNormalizer::new('▁', PrependMode::Unconditional, false)
        );
        let (fused, covered) = MetaspaceNormalizer::fuse(&steps[1..]).unwrap();
        assert_eq!(covered, 1);
        assert_eq!(
            fused,
            MetaspaceNormalizer::new('▁', PrependMode::Never, false)
        );
    }

    /// A chunk with nothing to swap is handed back without a copy, as `Replace` hands it back.
    #[test]
    fn a_swap_with_no_spaces_borrows() {
        let steps = [step(SWAP)];
        let (fused, _) = MetaspaceNormalizer::fuse(&steps).unwrap();
        assert!(matches!(
            pipeline::Normalizer::normalize(&fused, "no_spaces").unwrap(),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn refuses_a_replace_that_is_not_the_space_swap() {
        let refused = [
            // Swaps something else entirely; running it as the space swap would corrupt the text.
            (
                "another literal",
                r#"{"type":"Replace","pattern":{"String":"x"},"content":"y"}"#,
            ),
            // The delimiter is one char; two of them per space is not this normalizer's rewrite.
            (
                "multi-char content",
                r#"{"type":"Replace","pattern":{"String":" "},"content":"▁▁"}"#,
            ),
            // Deletes spaces instead of marking them.
            (
                "empty content",
                r#"{"type":"Replace","pattern":{"String":" "},"content":""}"#,
            ),
            // Only single spaces are swapped one-for-one.
            (
                "a two-space pattern",
                r#"{"type":"Replace","pattern":{"String":"  "},"content":"▁"}"#,
            ),
        ];
        for (name, json) in refused {
            assert!(MetaspaceNormalizer::fuse(&[step(json)]).is_none(), "{name}");
        }
    }

    /// A regex spelling is refused even when it happens to match only a space: fusing checks the
    /// pattern structurally and does not interpret regex syntax.
    #[cfg(feature = "fancy-regex")]
    #[test]
    fn refuses_a_regex_pattern_even_one_matching_a_space() {
        let json = r#"{"type":"Replace","pattern":{"Regex":" "},"content":"▁"}"#;
        assert!(MetaspaceNormalizer::fuse(&[step(json)]).is_none());
    }

    /// A `Prepend` writing anything but the swap's delimiter is a different rewrite; the pair must
    /// not fuse. The swap after it still can, on its own, once the caller reaches it.
    #[test]
    fn refuses_a_prepend_of_something_else() {
        for prepend in [
            r#"{"type":"Prepend","prepend":"x"}"#,
            r#"{"type":"Prepend","prepend":"▁▁"}"#,
        ] {
            let steps = [step(prepend), step(SWAP)];
            assert!(MetaspaceNormalizer::fuse(&steps).is_none(), "{prepend}");
            assert!(MetaspaceNormalizer::fuse(&steps[1..]).is_some());
        }
    }

    /// The pair only fuses when the two steps are adjacent: another step in between sees the text
    /// mid-rewrite, and fusing across it would change what that step sees.
    #[test]
    fn refuses_a_separated_pair() {
        let steps = [step(PREPEND), step(r#"{"type":"Lowercase"}"#), step(SWAP)];
        assert!(MetaspaceNormalizer::fuse(&steps).is_none());
        assert!(MetaspaceNormalizer::fuse(&steps[2..]).is_some());
    }
}
