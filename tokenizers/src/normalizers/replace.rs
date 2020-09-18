use crate::tokenizer::{NormalizedString, Normalizer, Result};
use serde::{Deserialize, Serialize};

/// This normalizer will take a `pattern` (for now only a String)
/// and replace every occurrence with `content`.
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(tag = "type")]
pub struct Replace {
    pattern: String,
    content: String,
}

impl Replace {
    pub fn new(pattern: String, content: String) -> Self {
        Self { pattern, content }
    }
}

impl Normalizer for Replace {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.replace(&self.pattern, &self.content)
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
        Replace::new("''".to_string(), "\"".to_string())
            .normalize(&mut n)
            .unwrap();

        assert_eq!(&n.get(), &normalized);
    }
}
