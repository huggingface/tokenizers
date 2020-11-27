use crate::tokenizer::{NormalizedString, Normalizer, Result};
use onig::Regex;
use serde::{Deserialize, Serialize};

/// Represents the different patterns that `Replace` can use
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReplacePattern {
    String(String),
    Regex(String),
}

impl From<String> for ReplacePattern {
    fn from(v: String) -> Self {
        ReplacePattern::String(v)
    }
}

impl From<&str> for ReplacePattern {
    fn from(v: &str) -> Self {
        ReplacePattern::String(v.to_owned())
    }
}

/// We use this custom deserializer to provide the value for `regex` for `Replace`
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
        Replace::new(v.pattern, v.content)
    }
}

/// This normalizer will take a `pattern` (for now only a String)
/// and replace every occurrence with `content`.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", try_from = "ReplaceDeserializer")]
pub struct Replace {
    pattern: ReplacePattern,
    content: String,
    #[serde(skip)]
    regex: Regex,
}

impl Clone for Replace {
    fn clone(&self) -> Self {
        Replace::new(self.pattern.clone(), &self.content).unwrap()
    }
}

impl PartialEq for Replace {
    fn eq(&self, other: &Replace) -> bool {
        self.pattern == other.pattern && self.content == other.content
    }
}

impl Replace {
    pub fn new<I: Into<ReplacePattern>, C: Into<String>>(pattern: I, content: C) -> Result<Self> {
        let pattern: ReplacePattern = pattern.into();
        let regex = match &pattern {
            ReplacePattern::String(s) => Regex::new(&regex::escape(s))?,
            ReplacePattern::Regex(r) => Regex::new(r)?,
        };

        Ok(Self {
            pattern,
            content: content.into(),
            regex,
        })
    }
}

impl Normalizer for Replace {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.replace(&self.regex, &self.content)
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

    #[test]
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
}
