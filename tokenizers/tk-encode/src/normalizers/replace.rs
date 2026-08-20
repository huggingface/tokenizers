use std::borrow::Cow;

use crate::pipeline;
use crate::utils::search::Search;

// `ReplacePattern` moved to `utils::search` with the matcher it configures; re-exported so the
// historical path keeps working.
use crate::tokenizer::{NormalizedString, Normalizer, Result};
pub use crate::utils::search::ReplacePattern;

/// This normalizer will take a `pattern` (for now only a String)
/// and replace every occurrence with `content`.
///
/// The on-disk shape lives in `tk-convert`'s `normalizers::mirror::replace`, which is also where
/// the `ReplaceDeserializer` + `TryFrom` pair went: `search` is derived from `pattern` by a
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

impl Normalizer for Replace {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        match &self.search {
            Search::Literal(literal) => normalized.replace(literal, &self.content),
            Search::Regex(regex) => normalized.replace(regex, &self.content),
            Search::Nothing => Ok(()),
        }
    }
}

/// Builds the text with every match swapped for `content`. Borrows the input back untouched when
/// nothing matched, which is what keeps the common case free of allocation.
fn replace_matches<'a>(
    input: &'a str,
    content: &str,
    matches: impl Iterator<Item = (usize, usize)>,
) -> Cow<'a, str> {
    let mut replaced: Option<String> = None;
    let mut last_end = 0;
    for (start, end) in matches {
        let replaced: &mut String =
            replaced.get_or_insert_with(|| String::with_capacity(input.len()));
        replaced.push_str(&input[last_end..start]);
        replaced.push_str(content);
        last_end = end;
    }
    match replaced {
        Some(mut replaced) => {
            replaced.push_str(&input[last_end..]);
            Cow::Owned(replaced)
        }
        None => Cow::Borrowed(input),
    }
}

impl pipeline::Normalizer for Replace {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
        Ok(match &self.search {
            Search::Literal(literal) => {
                let width = literal.pattern().len();
                let matches = literal
                    .matches(input.as_bytes())
                    .map(|start| (start, start + width));
                replace_matches(input, &self.content, matches)
            }
            Search::Regex(regex) => replace_matches(input, &self.content, regex.find_iter(input)),
            Search::Nothing => Cow::Borrowed(input),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replace() {
        let original = "This is a ''test''";
        let normalized = "This is a \"test\"";

        let mut n = NormalizedString::from(original);
        Replace::new("''", "\"").unwrap().normalize(&mut n).unwrap();

        assert_eq!(&n.get(), &normalized);
    }

    #[test]
    #[cfg(feature = "fancy-regex")] // a regex pattern needs a system-regex backend
    fn test_replace_regex() {
        let original = "This     is   a         test";
        let normalized = "This is a test";

        let mut n = NormalizedString::from(original);
        Replace::new(ReplacePattern::Regex(r"\s+".into()), ' ')
            .unwrap()
            .normalize(&mut n)
            .unwrap();

        assert_eq!(&n.get(), &normalized);
    }

    // `serialization` and `a_string_pattern_deserializes_with_no_backend` moved to
    // `tk-convert`'s `normalizers::mirror::tests` with the serde they exercise. What is left here
    // is everything that tests the normalizer itself rather than its on-disk shape.

    /// The goal of the literal path: a plain string pattern builds and runs with no regex backend.
    #[test]
    fn a_string_pattern_needs_no_backend() {
        let replace = Replace::new(" ", "▁").unwrap();
        assert_eq!(
            pipeline::Normalizer::normalize(&replace, "a b  c").unwrap(),
            "a▁b▁▁c"
        );
        // Nothing to replace: the input is handed back as it is, with nothing allocated.
        assert!(matches!(
            pipeline::Normalizer::normalize(&replace, "abc").unwrap(),
            Cow::Borrowed("abc")
        ));
        // An empty pattern would match everywhere, so it matches nowhere instead.
        let empty = Replace::new("", "x").unwrap();
        assert_eq!(
            pipeline::Normalizer::normalize(&empty, "abc").unwrap(),
            "abc"
        );
    }

    fn assert_pipeline_matches_legacy(n: &Replace, inputs: &[&str]) {
        for input in inputs {
            let mut ns = NormalizedString::from(*input);
            Normalizer::normalize(n, &mut ns).unwrap(); // legacy oracle
            assert_eq!(
                ns.get(),
                &*pipeline::Normalizer::normalize(n, input).unwrap()
            );
        }
    }

    #[test]
    fn pipeline_replace_matches_legacy() {
        let n = Replace::new("''", "\"").unwrap();
        assert_pipeline_matches_legacy(&n, &["This is a ''test''", "no quotes", ""]);
    }

    #[test]
    #[cfg(feature = "fancy-regex")] // a regex pattern needs a system-regex backend
    fn pipeline_replace_matches_legacy_for_a_regex() {
        let n = Replace::new(ReplacePattern::Regex(r"\s+".into()), " ").unwrap();
        assert_pipeline_matches_legacy(&n, &["a   b   c", "single", ""]);
    }
}
