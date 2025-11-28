use crate::tokenizer::{NormalizedString, Normalizer, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
pub struct Append {
    pub append: String,
}

impl Append {
    pub fn new(append: String) -> Self {
        Self { append }
    }
}

impl Normalizer for Append {
    /// Append the normalized string inplace
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        if !normalized.is_empty() {
            normalized.append(&self.append);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_append() {
        let original = "Hello";
        let normalized = "Hello▁";
        assert_ne!(original, normalized);
        let mut n = NormalizedString::from(original);
        let append = Append::new("▁".to_string());
        append.normalize(&mut n).unwrap();
        assert_eq!(&n.get(), &normalized);
    }
}
