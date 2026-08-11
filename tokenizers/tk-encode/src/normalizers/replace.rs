use std::borrow::Cow;

use crate::pipeline;
use crate::tokenizer::Decoder;
use crate::tokenizer::pattern::Pattern;
use crate::tokenizer::{Normalizer, Result};
use crate::utils::SysRegex;
use atomsplit::literal::Literal;
use serde::{Deserialize, Serialize};

/// Represents the different patterns that `Replace` can use
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Eq)]
pub enum ReplacePattern {
    String(String),
    Regex(String),
}

impl From<String> for ReplacePattern {
    fn from(v: String) -> Self {
        Self::String(v)
    }
}

impl From<&str> for ReplacePattern {
    fn from(v: &str) -> Self {
        Self::String(v.to_owned())
    }
}

/// We use this custom deserializer to build the search for `Replace`
#[doc(hidden)]
#[derive(Deserialize)]
#[serde(tag = "type")]
struct ReplaceDeserializer {
    pattern: ReplacePattern,
    content: String,
}

impl std::convert::TryFrom<ReplaceDeserializer> for Replace {
    type Error = Box<dyn std::error::Error + Send + Sync>;

    fn try_from(v: ReplaceDeserializer) -> Result<Self> {
        Self::new(v.pattern, v.content)
    }
}

/// How a [`Replace`] looks for its pattern.
#[derive(Debug)]
enum Search {
    /// A plain string, scanned for directly — no regex engine involved, so this works in every build.
    Literal(Literal),
    /// A real regex, which needs the system backend (the `fancy-regex` feature).
    Regex(SysRegex),
    /// The empty string, which matches nothing.
    Nothing,
}

impl Search {
    fn find_matches(&self, inside: &str) -> Result<Vec<((usize, usize), bool)>> {
        match self {
            Self::Literal(literal) => literal.find_matches(inside),
            Self::Regex(regex) => regex.find_matches(inside),
            Self::Nothing => Ok(vec![((0, inside.len()), false)]),
        }
    }
}

/// This normalizer will take a `pattern` (for now only a String)
/// and replace every occurrence with `content`.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", try_from = "ReplaceDeserializer")]
pub struct Replace {
    pattern: ReplacePattern,
    pub content: String,
    #[serde(skip)]
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
        let search = match &pattern {
            ReplacePattern::String(s) if s.is_empty() => Search::Nothing,
            ReplacePattern::String(s) => Search::Literal(Literal::new(s.as_bytes())?),
            ReplacePattern::Regex(r) => Search::Regex(SysRegex::new(r)?),
        };

        Ok(Self {
            pattern,
            content: content.into(),
            search,
        })
    }
}

impl Normalizer for Replace {}

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

impl Decoder for Replace {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        tokens
            .into_iter()
            .map(|token| -> Result<String> {
                let mut new_token = "".to_string();

                for ((start, stop), is_match) in self.search.find_matches(&token)? {
                    if is_match {
                        new_token.push_str(&self.content);
                    } else {
                        new_token.push_str(&token[start..stop]);
                    }
                }
                Ok(new_token)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalizers::assert_normalizes;

    #[test]
    fn replace_swaps_every_occurrence() {
        assert_normalizes(
            &Replace::new("''", "\"").unwrap(),
            &[
                ("This is a ''test''", "This is a \"test\""),
                ("no quotes", "no quotes"),
                ("", ""),
            ],
        );
    }

    #[test]
    #[cfg(feature = "fancy-regex")] // a regex pattern needs a system-regex backend
    fn replace_swaps_every_regex_match() {
        assert_normalizes(
            &Replace::new(ReplacePattern::Regex(r"\s+".into()), " ").unwrap(),
            &[
                ("This     is   a         test", "This is a test"),
                ("a   b   c", "a b c"),
                ("single", "single"),
                ("", ""),
            ],
        );
    }

    #[test]
    #[cfg(feature = "fancy-regex")] // the regex half of this needs a system-regex backend
    fn serialization() {
        let replace = Replace::new("Hello", "Hey").unwrap();
        let replace_s = r#"{"type":"Replace","pattern":{"String":"Hello"},"content":"Hey"}"#;
        assert_eq!(serde_json::to_string(&replace).unwrap(), replace_s);
        assert_eq!(serde_json::from_str::<Replace>(replace_s).unwrap(), replace);

        let replace = Replace::new(ReplacePattern::Regex(r"\s+".into()), ' ').unwrap();
        let replace_s = r#"{"type":"Replace","pattern":{"Regex":"\\s+"},"content":" "}"#;
        assert_eq!(serde_json::to_string(&replace).unwrap(), replace_s);
        assert_eq!(serde_json::from_str::<Replace>(replace_s).unwrap(), replace);
    }

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

    /// A config spelling its pattern as a string must also *deserialize* with no backend — the
    /// regex half of `serialization` above can only run once one is compiled.
    #[test]
    fn a_string_pattern_deserializes_with_no_backend() {
        let replace_s = r#"{"type":"Replace","pattern":{"String":"Hello"},"content":"Hey"}"#;
        let replace = Replace::new("Hello", "Hey").unwrap();
        assert_eq!(serde_json::from_str::<Replace>(replace_s).unwrap(), replace);
        assert_eq!(serde_json::to_string(&replace).unwrap(), replace_s);
    }

    #[test]
    fn test_replace_decode() {
        let original = vec!["hello".to_string(), "_hello".to_string()];
        let replace = Replace::new("_", " ").unwrap();
        assert_eq!(
            replace.decode_chain(original).unwrap(),
            vec!["hello", " hello"]
        );
    }
}
