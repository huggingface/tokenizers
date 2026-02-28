use crate::tokenizer::pattern::Pattern;
use crate::tokenizer::Decoder;
use crate::tokenizer::{NormalizedString, Normalizer, Result};
use crate::utils::SysRegex;
use regex::Regex;
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

/// Converts backreferences in the regex pattern string
fn convert_backrefs(pattern: &str) -> String {
    let re = Regex::new(r"\\(\d+)").unwrap(); // match \1, \2, etc.
    let converted = re.replace_all(pattern, |caps: &regex::Captures| {
        format!("${{{}}}", &caps[1]) // insert the captured number dynamically
    }).to_string();
    converted
}


impl Replace {
    pub fn new<I: Into<ReplacePattern>, C: Into<String>>(pattern: I, content: C) -> Result<Self> {
        let pattern: ReplacePattern = pattern.into();
        let converted_pattern = match &pattern {
            ReplacePattern::String(s) => SysRegex::new(&regex::escape(s))?,
            ReplacePattern::Regex(r) => SysRegex::new(r)?,
        };

        let converted_content = convert_backrefs(&content.into()); // Apply convert_backrefs to content

        Ok(Self {
            pattern,
            content: converted_content,
            regex: converted_pattern,
        })
    }
}

impl Normalizer for Replace {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        match &self.pattern {
            ReplacePattern::Regex(pattern) => {
                // Use the regex pattern directly for replacement
                let re = Regex::new(pattern)?;
                let current_text = normalized.get().to_owned();
                let result = re.replace_all(&current_text, &self.content);

                // Directly set the normalized string to the result
                *normalized = NormalizedString::from(result.as_ref());
                Ok(())
            }
            ReplacePattern::String(_) => {
                // Handle simple string replacement
                normalized.replace(&self.regex, &self.content)
            }
        }
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
    fn test_replace_with_capture_groups() {
        // Test case 1: Simple capture group and backreference
        let text = "le travail est totalement pénible";
        let normalizer = Replace::new(ReplacePattern::Regex(r"(l)(e)".into()), r"\1 \2").unwrap();
        let mut normalized = NormalizedString::from(text);
        normalizer.normalize(&mut normalized).unwrap();
        let output = normalized.get();
        println!("output is: {}", output);
        assert!(output.contains("l e travail est total ement pénibl e"));
    
        // Test case 2: Phone number formatting
        let text = "123-456-7890";
        let normalizer = Replace::new(
            ReplacePattern::Regex(r"(\d{3})-(\d{3})-(\d{4})".into()),
            r"(\1) \2-\3"
        ).unwrap();
        let mut normalized = NormalizedString::from(text);
        normalizer.normalize(&mut normalized).unwrap();
        let output = normalized.get();
        println!("output is111: {}", output);
        assert!(output.contains("(123) 456-7890"));
    
        // Test case 3: Greedy matching of repeated characters
        let text = "aaaabbbbcccc";
        let normalizer = Replace::new(
            ReplacePattern::Regex(r"(a+)(b+)(c+)".into()),
            r"[$1]-[$2]-[$3]"
        ).unwrap();
        let mut normalized = NormalizedString::from(text);
        normalizer.normalize(&mut normalized).unwrap();
        assert!(normalized.get().contains("[aaaa]-[bbbb]-[cccc]"));
    
        // Test case 4: Non-greedy match with wildcards
        let text = "<p>Some text</p><p>More text</p>";
        let normalizer = Replace::new(
            ReplacePattern::Regex(r"<p>(.*?)</p>".into()),
            r"[P:$1]"
        ).unwrap();
        let mut normalized = NormalizedString::from(text);
        normalizer.normalize(&mut normalized).unwrap();
        assert!(normalized.get().contains("[P:Some text]"));
        assert!(normalized.get().contains("[P:More text]"));
    
        // Test case 5: Unicode capture and replace
        let text = "東京 is the capital of 日本";
        let normalizer = Replace::new(
            ReplacePattern::Regex(r"(東京)".into()),
            r"$1 (Tokyo)"
        ).unwrap();
        let mut normalized = NormalizedString::from(text);
        normalizer.normalize(&mut normalized).unwrap();
        assert!(normalized.get().contains("東京 (Tokyo)"));
    
        // Test case 6: Backreferences with slashes and quotes
        let text = "name=\"value\"";
        let normalizer = Replace::new(
            ReplacePattern::Regex(r#"name="([^"]+)""#.into()),
            r"name='$1'"
        ).unwrap();
        let mut normalized = NormalizedString::from(text);
        normalizer.normalize(&mut normalized).unwrap();
        assert!(normalized.get().contains("name='value'"));

        // Test case 7: Replace dollar sign with USD prefix
        let text = "Price is $20 and discounted to $15";
        let normalizer = Replace::new(
            ReplacePattern::Regex(r"\$(\d+)".into()),
            r"USD $1"
        ).unwrap();
        let mut normalized = NormalizedString::from(text);
        normalizer.normalize(&mut normalized).unwrap();
        assert!(normalized.get().contains("USD 20"));
        assert!(normalized.get().contains("USD 15"));

        // Test case 8: Escape literal dollar signs
        let text = "Cost: $5, Tax: $0.50";
        let normalizer = Replace::new(
            ReplacePattern::Regex(r"\$".into()),
            r"\\$"
        ).unwrap();
        let mut normalized = NormalizedString::from(text);
        normalizer.normalize(&mut normalized).unwrap();
        assert!(normalized.get().contains("\\$5"));
        assert!(normalized.get().contains("\\$0.50"));

        // Test case 9: Replace dollars with euros
        let text = "Item costs $40, not $50";
        let normalizer = Replace::new(
            ReplacePattern::Regex(r"\$(\d+)".into()),
            r"€$1"
        ).unwrap();
        let mut normalized = NormalizedString::from(text);
        normalizer.normalize(&mut normalized).unwrap();
        assert!(normalized.get().contains("€40"));
        assert!(normalized.get().contains("€50"));

        // Test case 10: Keep dollar amount but add comma formatting
        let text = "That car costs $1000000!";
        let normalizer = Replace::new(
            ReplacePattern::Regex(r"\$(\d{1,3})(\d{3})(\d{3})".into()),
            r"$$1,$2,$3"
        ).unwrap();
        let mut normalized = NormalizedString::from(text);
        normalizer.normalize(&mut normalized).unwrap();
        assert!(normalized.get().contains("$1,000,000"));

        // Test case 11: Dollar sign at end of line (should not match as regex end anchor)
        let text = "I love making $";
        let normalizer = Replace::new(
            ReplacePattern::Regex(r"\$(\d*)".into()),
            r"USD\1"
        ).unwrap();
        let mut normalized = NormalizedString::from(text);
        normalizer.normalize(&mut normalized).unwrap();
        assert!(normalized.get().contains("USD"));
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
}
