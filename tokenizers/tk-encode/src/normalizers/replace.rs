use std::borrow::Cow;

use crate::pipeline;
use crate::tokenizer::pattern::Search;

use crate::tokenizer::Result;
// `ReplacePattern` lives in `tokenizer::pattern` with the `Search` it configures. This keeps its
// public path.
pub use crate::tokenizer::pattern::ReplacePattern;

/// This normalizer will take a `pattern` (for now only a String)
/// and replace every occurrence with `content`.
///
/// The on-disk shape is in [`super::serialization`]: `search` is derived from `pattern` by a
/// constructor that can fail, so a config can only ever be turned into a `Replace` through
/// `Replace::new`, never field by field.
#[derive(Debug)]
pub struct Replace {
    pattern: ReplacePattern,
    pub content: String,
    search: Search,
}

impl Clone for Replace {
    fn clone(&self) -> Self {
        Self::new(self.pattern.clone(), &self.content).unwrap()
    }
}

impl PartialEq for Replace {
    fn eq(&self, other: &Self) -> bool {
        self.pattern == other.pattern && self.content == other.content
    }
}

impl Replace {
    pub fn new<I: Into<ReplacePattern>, C: Into<String>>(pattern: I, content: C) -> Result<Self> {
        let pattern: ReplacePattern = pattern.into();
        let search = Search::new(&pattern)?;

        Ok(Self {
            pattern,
            content: content.into(),
            search,
        })
    }
}

impl Replace {
    /// The pattern as written in the config. Needed to write it back out.
    pub fn pattern(&self) -> &ReplacePattern {
        &self.pattern
    }
}

impl pipeline::Normalizer for Replace {
    fn normalize<'a>(&self, input: &'a str, _offset: usize) -> Result<Cow<'a, str>> {
        self.search.replace(input, &self.content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The two JSON round-trip tests went with `super::serialization` and the serde they exercised.
    // What is left here tests the normalizer itself rather than its on-disk shape.

    // Expected values were captured from the legacy `NormalizedString` normalizer this compared
    // against, on the commit that removed it, so they still pin what the two agreed on.
    fn assert_pipeline(n: &Replace, cases: &[(&str, &str)]) {
        for &(input, expected) in cases {
            assert_eq!(
                &*pipeline::Normalizer::normalize(n, input, 0).unwrap(),
                expected,
                "input={input:?}"
            );
        }
    }

    #[test]
    fn pipeline_replace() {
        let n = Replace::new("''", "\"").unwrap();
        assert_pipeline(
            &n,
            &[
                ("This is a ''test''", "This is a \"test\""),
                ("no quotes", "no quotes"),
                ("", ""),
            ],
        );
    }

    #[test]
    #[cfg(feature = "fancy-regex")] // a regex pattern needs a system-regex backend
    fn pipeline_replace_for_a_regex() {
        let n = Replace::new(ReplacePattern::Regex(r"\s+".into()), " ").unwrap();
        assert_pipeline(
            &n,
            &[("a   b   c", "a b c"), ("single", "single"), ("", "")],
        );
    }

    #[test]
    #[cfg(feature = "fancy-regex")] // a regex pattern needs a system-regex backend
    fn refuses_a_regex_match_overlapping_the_previous_one() {
        let n = Replace::new(ReplacePattern::Regex(r"(?<=\K.).|.".into()), "-").unwrap();
        assert!(pipeline::Normalizer::normalize(&n, "Hello", 0).is_err());
    }
}
