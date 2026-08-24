use std::borrow::Cow;

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
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
        let s = if self.strip_left {
            input.trim_start()
        } else {
            input
        };
        let s = if self.strip_right { s.trim_end() } else { s };
        Ok(Cow::Borrowed(s))
    }
}

// This normalizer removes combining marks from a normalized string
// It's different from unidecode as it does not attempt to modify
// non ascii languages.
#[cfg(feature = "normalizers")]
#[derive(Copy, Clone, Debug)]
pub struct StripAccents;

#[cfg(feature = "normalizers")]
impl pipeline::Normalizer for StripAccents {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
        if input.chars().any(is_combining_mark) {
            Ok(Cow::Owned(
                input.chars().filter(|&c| !is_combining_mark(c)).collect(),
            ))
        } else {
            Ok(Cow::Borrowed(input))
        }
    }
}

#[cfg(all(test, feature = "normalizers"))]
mod tests {
    use super::*;

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
                &*pipeline::Normalizer::normalize(&n, input).unwrap(),
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
                    &*pipeline::Normalizer::normalize(&n, input).unwrap(),
                    expected,
                    "strip=({strip_left}, {strip_right}) input={input:?}"
                );
            }
        }
    }
}
