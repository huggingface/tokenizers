use crate::tokenizer::{NormalizedString, Normalizer, Result};
use compact_str::CompactString;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
pub struct Prepend {
    pub prepend: CompactString,
}

impl Prepend {
    pub fn new(prepend: impl Into<CompactString>) -> Self {
        Self {
            prepend: prepend.into(),
        }
    }
}

impl Normalizer for Prepend {
    /// Strip the normalized string inplace
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        if !normalized.is_empty() {
            normalized.prepend(&self.prepend);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prepend() {
        let original = "Hello";
        let normalized = "▁Hello";
        assert_ne!(original, normalized);
        let mut n = NormalizedString::from(original);
        let prepend = Prepend::new("▁".to_string());
        prepend.normalize(&mut n).unwrap();
        assert_eq!(&n.get(), &normalized);
        assert_eq!(
            n,
            NormalizedString::new(
                original.to_string(),
                normalized.to_string(),
                vec![
                    (0, 1),
                    (0, 1),
                    (0, 1),
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 4),
                    (4, 5)
                ],
                0
            )
        );
        assert_eq!(
            n.alignments_original(),
            vec![(0, 4), (4, 5), (5, 6), (6, 7), (7, 8)]
        );
    }
}
