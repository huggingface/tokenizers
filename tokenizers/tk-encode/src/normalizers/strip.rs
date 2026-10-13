use std::borrow::Cow;
#[cfg(feature = "normalizers")]
use std::sync::LazyLock;

#[cfg(feature = "normalizers")]
use crate::normalizers::utils::{CharacterCache, Rewriter, unstable_spans_iter};
use crate::pipeline;
use crate::tokenizer::Result;
#[cfg(feature = "normalizers")]
use unicode_normalization_alignments::char::is_combining_mark;

/// Both fields are required, which is the *only* thing that rejects a tag-less object here.
#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub struct Strip {
    pub strip_left: bool,
    pub strip_right: bool,
}

impl Strip {
    pub fn new(strip_left: bool, strip_right: bool) -> Self {
        Self {
            strip_left,
            strip_right,
        }
    }
}

impl pipeline::Normalizer for Strip {
    fn normalize<'a>(&self, input: &'a str, _offset: usize) -> Result<Cow<'a, str>> {
        let s = if self.strip_left {
            input.trim_start()
        } else {
            input
        };
        let s = if self.strip_right { s.trim_end() } else { s };
        Ok(Cow::Borrowed(s))
    }
}

#[cfg(feature = "normalizers")]
static STRIP_ACCENTS_CACHE: LazyLock<CharacterCache> =
    LazyLock::new(|| CharacterCache::new(is_combining_mark));

// This normalizer removes combining marks from a normalized string
// It's different from unidecode as it does not attempt to modify
// non ascii languages.
#[cfg(feature = "normalizers")]
#[derive(Copy, Clone, Debug)]
pub struct StripAccents;

#[cfg(feature = "normalizers")]
impl pipeline::Normalizer for StripAccents {
    fn normalize<'a>(&self, input: &'a str, _offset: usize) -> Result<Cow<'a, str>> {
        let mut rewriter = Rewriter::new(input);
        for span in unstable_spans_iter(input, &STRIP_ACCENTS_CACHE, false) {
            rewriter.rewrite(span); // write nothing = delete
        }
        Ok(rewriter.finish())
    }
}

#[cfg(all(test, feature = "normalizers"))]
mod tests {
    use super::*;
    use crate::normalizers::utils::random_texts;

    /// Expected values were captured from the legacy `NormalizedString` normalizer this test used
    /// to compare against, on the commit that removed it -- so they still pin exactly what the two
    /// implementations agreed on, without keeping the legacy code alive to ask.
    #[test]
    fn pipeline_strip_accents() {
        let n = StripAccents;
        for (input, expected) in [
            ("café", "café"),
            ("abc", "abc"),
            ("", ""),
            ("å ç ñ", "å ç ñ"),
            ("     hello", "     hello"),
        ] {
            assert_eq!(
                &*pipeline::Normalizer::normalize(&n, input, 0).unwrap(),
                expected,
                "input={input:?}"
            );
        }
    }

    /// Expected values were captured from the legacy `NormalizedString` normalizer this test used
    /// to compare against, on the commit that removed it -- so they still pin exactly what the two
    /// implementations agreed on, without keeping the legacy code alive to ask.
    #[test]
    fn pipeline_strip() {
        #[allow(clippy::type_complexity)]
        let cases: &[((bool, bool), &[(&str, &str)])] = &[
            (
                (true, true),
                &[
                    ("  hello  ", "hello"),
                    ("hello", "hello"),
                    ("", ""),
                    ("   ", ""),
                    ("\t hi \n", "hi"),
                ][..],
            ),
            (
                (true, false),
                &[
                    ("  hello  ", "hello  "),
                    ("hello", "hello"),
                    ("", ""),
                    ("   ", ""),
                    ("\t hi \n", "hi \n"),
                ][..],
            ),
            (
                (false, true),
                &[
                    ("  hello  ", "  hello"),
                    ("hello", "hello"),
                    ("", ""),
                    ("   ", ""),
                    ("\t hi \n", "\t hi"),
                ][..],
            ),
        ];
        for &((strip_left, strip_right), pairs) in cases {
            let n = Strip::new(strip_left, strip_right);
            for &(input, expected) in pairs {
                assert_eq!(
                    &*pipeline::Normalizer::normalize(&n, input, 0).unwrap(),
                    expected,
                    "strip=({strip_left}, {strip_right}) input={input:?}"
                );
            }
        }
    }

    #[test]
    fn strip_accents_matches_the_char_filter() {
        let pieces = [
            "e",
            "\u{301}",
            "\u{323}\u{301}",
            "\u{e9}",
            "e\u{301}",
            "abcdefghijklmnopqrstuvwxyz",
            " ",
            "\u{416}",
            "\u{5d0}\u{5b0}",
            "\u{915}\u{93c}\u{94d}",
            "\u{1d165}\u{1d16e}",
            "\u{4e2d}",
            "\u{1f600}",
        ];
        for text in random_texts(3, &pieces, 1000, 120) {
            let expected: String = text.chars().filter(|&c| !is_combining_mark(c)).collect();
            let got = pipeline::Normalizer::normalize(&StripAccents, &text, 0).unwrap();
            assert_eq!(got, expected, "{text:?}");
        }
    }

    #[test]
    fn strip_accents_borrows_text_without_marks() {
        for text in [
            "",
            "abc",
            "caf\u{e9}",
            "\u{416}\u{436} \u{4e2d}",
            &"x".repeat(40),
        ] {
            let got = pipeline::Normalizer::normalize(&StripAccents, text, 0).unwrap();
            assert!(matches!(got, Cow::Borrowed(_)), "{text:?}");
        }
    }
}
