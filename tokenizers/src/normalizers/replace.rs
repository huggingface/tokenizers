use crate::tokenizer::pattern::Pattern;
use crate::tokenizer::Decoder;
use crate::tokenizer::{NormalizedString, Normalizer, Result};
use crate::utils::SysRegex;
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
        Self::new(v.pattern, v.content)
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
    regex: SysRegex,
    #[serde(skip)]
    expansion_re: Option<regex::Regex>,
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
        let regex = match &pattern {
            ReplacePattern::String(s) => SysRegex::new(&regex::escape(s))?,
            ReplacePattern::Regex(r) => SysRegex::new(r)?,
        };
        let expansion_re = match &pattern {
            ReplacePattern::String(_) => None,
            ReplacePattern::Regex(r) => regex::Regex::new(r).ok(),
        };

        Ok(Self {
            pattern,
            content: content.into(),
            regex,
            expansion_re,
        })
    }
}

impl Normalizer for Replace {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.replace_regex(&self.regex, self.expansion_re.as_ref(), &self.content)
    }
}

impl Decoder for Replace {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        match &self.expansion_re {
            Some(re) => tokens
                .into_iter()
                .map(|token| Ok(re.replace_all(&token, &self.content).to_string()))
                .collect(),
            None => tokens
                .into_iter()
                .map(|token| {
                    let mut new_token = String::new();
                    for ((start, stop), is_match) in (&self.regex).find_matches(&token)? {
                        if is_match {
                            new_token.push_str(&self.content);
                        } else {
                            new_token.push_str(&token[start..stop]);
                        }
                    }
                    Ok(new_token)
                })
                .collect(),
        }
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

    #[test]
    fn test_replace_decode() {
        let original = vec!["hello".to_string(), "_hello".to_string()];
        let replace = Replace::new("_", " ").unwrap();
        assert_eq!(
            replace.decode_chain(original).unwrap(),
            vec!["hello", " hello"]
        );
    }

    #[test]
    fn test_replace_regex_groups_dollar() {
        let original = "le travail";
        let expected = "l e travail";

        let mut n = NormalizedString::from(original);
        Replace::new(ReplacePattern::Regex(r"(l)(e)".into()), r"$1 $2")
            .unwrap()
            .normalize(&mut n)
            .unwrap();

        assert_eq!(&n.get(), &expected);
    }

    #[test]
    fn test_replace_regex_groups_decode() {
        let tokens = vec!["le".to_string(), "test".to_string()];
        let replace = Replace::new(ReplacePattern::Regex(r"(l)(e)".into()), r"$1 $2").unwrap();
        assert_eq!(
            replace.decode_chain(tokens).unwrap(),
            vec!["l e", "test"]
        );
    }
}
