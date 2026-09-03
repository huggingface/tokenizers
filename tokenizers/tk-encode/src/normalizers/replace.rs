use std::borrow::Cow;

use crate::pipeline;
use crate::utils::search::Search;

// `ReplacePattern` moved to `utils::search` with the matcher it configures; re-exported so the
// historical path keeps working.
use crate::tokenizer::Result;
pub use crate::utils::search::ReplacePattern;

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

    // The two JSON round-trip tests went with `super::serialization` and the serde they exercised.
    // What is left here tests the normalizer itself rather than its on-disk shape.

    // Expected values were captured from the legacy `NormalizedString` normalizer this compared
    // against, on the commit that removed it, so they still pin what the two agreed on.
    fn assert_pipeline(n: &Replace, cases: &[(&str, &str)]) {
        for &(input, expected) in cases {
            assert_eq!(
                &*pipeline::Normalizer::normalize(n, input).unwrap(),
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
}
