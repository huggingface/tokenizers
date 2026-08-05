use std::borrow::Cow;

use crate::pipeline;
use crate::tokenizer::{NormalizedString, Normalizer, Result};
#[cfg(feature = "config")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "config", derive(Deserialize, Serialize))]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "config", serde(tag = "type"))]
pub struct Prepend {
    pub prepend: String,
}

impl Prepend {
    pub fn new(prepend: String) -> Self {
        Self { prepend }
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

impl pipeline::Normalizer for Prepend {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
        if input.is_empty() || self.prepend.is_empty() {
            return Ok(Cow::Borrowed(input));
        }
        Ok(Cow::Owned(format!(
            "{prepend}{input}",
            prepend = self.prepend
        )))
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

    #[test]
    fn pipeline_prepend_matches_legacy() {
        let n = Prepend::new("▁".to_string());
        for input in &["Hello", "world", ""] {
            let mut ns = NormalizedString::from(*input);
            Normalizer::normalize(&n, &mut ns).unwrap(); // legacy oracle
            assert_eq!(
                ns.get(),
                &*pipeline::Normalizer::normalize(&n, input).unwrap()
            );
        }
    }
}
