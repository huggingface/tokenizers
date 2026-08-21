//! The `Replace` **decoder**.
//!
//! A different type from the `Replace` *normalizer*, deliberately. They read the same JSON and do
//! the same substitution, but a decoder and a normalizer are different roles, and having one type
//! implement both is what used to make `DecoderWrapper` and `NormalizerWrapper` share a variant —
//! which in turn meant one type had to answer for both on-disk shapes. One type, one
//! role: the shared part is the matcher in [`crate::utils::search`], not the component.

use crate::tokenizer::{Decoder, Result};
use crate::utils::search::{ReplacePattern, Search};

/// Replaces every occurrence of `pattern` with `content` in each token being decoded.
#[derive(Debug)]
pub struct ReplaceDecoder {
    pattern: ReplacePattern,
    content: String,
    search: Search,
}

impl ReplaceDecoder {
    pub fn new<I: Into<ReplacePattern>, C: Into<String>>(pattern: I, content: C) -> Result<Self> {
        let pattern: ReplacePattern = pattern.into();
        let search = Search::new(&pattern)?;
        Ok(Self {
            pattern,
            content: content.into(),
            search,
        })
    }

    /// The pattern as written in the config. Needed to write it back out.
    pub fn pattern(&self) -> &ReplacePattern {
        &self.pattern
    }

    pub fn content(&self) -> &str {
        &self.content
    }
}

// `search` is derived from `pattern`, so it is rebuilt rather than cloned, and ignored when
// comparing: two `ReplaceDecoder`s with the same pattern and content are the same decoder.
impl Clone for ReplaceDecoder {
    fn clone(&self) -> Self {
        Self::new(self.pattern.clone(), &self.content).unwrap()
    }
}

impl PartialEq for ReplaceDecoder {
    fn eq(&self, other: &Self) -> bool {
        self.pattern == other.pattern && self.content == other.content
    }
}

impl Decoder for ReplaceDecoder {
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

    #[test]
    fn decodes_by_replacing_each_match() {
        let original = vec!["hello".to_string(), "_hello".to_string()];
        let replace = ReplaceDecoder::new("_", " ").unwrap();
        assert_eq!(
            replace.decode_chain(original).unwrap(),
            vec!["hello", " hello"]
        );
    }

    /// `search` is derived from `pattern`, so it must not participate in either.
    #[test]
    fn clone_and_eq_ignore_the_derived_matcher() {
        let a = ReplaceDecoder::new("_", " ").unwrap();
        assert_eq!(a.clone(), a);
        assert_ne!(a, ReplaceDecoder::new("-", " ").unwrap());
    }
}
