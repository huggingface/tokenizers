use std::borrow::Cow;

use crate::pipeline;
use crate::tokenizer::Decoder;
use crate::tokenizer::pattern::Pattern;
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

impl pipeline::Normalizer for Replace {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
        let iter = self.regex.find_iter(input);
        let mut replaced: Option<String> = None;
        let mut last_end = 0;

        for (start, end) in iter {
            let replaced: &mut String =
                replaced.get_or_insert_with(|| String::with_capacity(input.len()));
            replaced.push_str(&input[last_end..start]);
            replaced.push_str(&self.content);
            last_end = end;
        }
        if let Some(mut replaced) = replaced {
            if last_end < input.len() {
                replaced.push_str(&input[last_end..]);
            }
            return Ok(Cow::Owned(replaced));
        }
        Ok(input.into())
    }
}

impl Decoder for Replace {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        tokens
            .into_iter()
            .map(|token| -> Result<String> {
                let mut new_token = "".to_string();

                for ((start, stop), is_match) in (&self.regex).find_matches(&token)? {
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

impl pipeline::Decoder for Replace {
    fn decode_token(
        &self,
        _state: &mut pipeline::DecoderState,
        _token_id: u32,
        token_bytes: &[u8],
        decoded: &mut Vec<u8>,
    ) -> Result<()> {
        match &self.pattern {
            // Plain byte search: no regex machinery on the hot path, and raw
            // (non-UTF-8) token bytes pass through unharmed.
            ReplacePattern::String(pattern) if !pattern.is_empty() => {
                let pat = pattern.as_bytes();
                let mut rest = token_bytes;
                while let Some(pos) = rest.windows(pat.len()).position(|window| window == pat) {
                    decoded.extend_from_slice(&rest[..pos]);
                    decoded.extend_from_slice(self.content.as_bytes());
                    rest = &rest[pos + pat.len()..];
                }
                decoded.extend_from_slice(rest);
                Ok(())
            }
            _ => {
                let token = std::str::from_utf8(token_bytes).map_err(
                    |_| "Replace decoder with a regex pattern requires valid UTF-8 tokens",
                )?;
                let mut last_end = 0;
                for (start, end) in self.regex.find_iter(token) {
                    decoded.extend_from_slice(&token.as_bytes()[last_end..start]);
                    decoded.extend_from_slice(self.content.as_bytes());
                    last_end = end;
                }
                decoded.extend_from_slice(&token.as_bytes()[last_end..]);
                Ok(())
            }
        }
    }
}

// `Replace` needs a system-regex backend (SysRegex) for every test here.
#[cfg(all(test, feature = "fancy-regex"))]
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
    fn pipeline_decode_token_matches_decode_chain() {
        let cases = vec![
            (
                Replace::new("▁", " ").unwrap(),
                vec!["▁Hey", "▁▁friend", "no_meta", "▁", ""],
            ),
            (Replace::new("ab", "X").unwrap(), vec!["aabb", "abab", "b"]),
            (
                Replace::new(ReplacePattern::Regex(r"\s+".into()), " ").unwrap(),
                vec!["a   b", " x ", "y"],
            ),
        ];
        for (replace, tokens) in cases {
            let expected = replace
                .decode_chain(tokens.iter().map(|t| t.to_string()).collect())
                .unwrap()
                .concat();
            let mut state = pipeline::DecoderState::default();
            let mut out = Vec::new();
            for token in &tokens {
                pipeline::Decoder::decode_token(
                    &replace,
                    &mut state,
                    0,
                    token.as_bytes(),
                    &mut out,
                )
                .unwrap();
            }
            assert_eq!(out, expected.as_bytes(), "pattern {:?}", replace.pattern);
        }
    }

    #[test]
    fn pipeline_replace_matches_legacy() {
        let n = Replace::new("''", "\"").unwrap();
        for input in &["This is a ''test''", "no quotes", ""] {
            let mut ns = NormalizedString::from(*input);
            Normalizer::normalize(&n, &mut ns).unwrap(); // legacy oracle
            assert_eq!(
                ns.get(),
                &*pipeline::Normalizer::normalize(&n, input).unwrap()
            );
        }

        let n = Replace::new(ReplacePattern::Regex(r"\s+".into()), " ").unwrap();
        for input in &["a   b   c", "single", ""] {
            let mut ns = NormalizedString::from(*input);
            Normalizer::normalize(&n, &mut ns).unwrap(); // legacy oracle
            assert_eq!(
                ns.get(),
                &*pipeline::Normalizer::normalize(&n, input).unwrap()
            );
        }
    }
}
