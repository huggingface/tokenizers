use std::borrow::Cow;

use crate::pipeline;
use crate::tokenizer::{NormalizedString, Normalizer, Result};
#[cfg(feature = "serde")]
use crate::utils::macro_rules_attribute;

// The `Sequence` normalizer is a `Vec<NormalizerWrapper>`, so it lives in `tk-convert` with
// the wrapper it is parameterised by.

/// Lowercases the input
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", macro_rules_attribute(impl_serde_type!))]
pub struct Lowercase;
impl Normalizer for Lowercase {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.lowercase();
        Ok(())
    }
}

/// Whether lowercasing `c` leaves it unchanged (a single, identical char)
pub(crate) fn lowercases_to_self(c: char) -> bool {
    let mut it = c.to_lowercase();
    matches!((it.next(), it.next()), (Some(first), None) if first == c)
}

impl pipeline::Normalizer for Lowercase {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
        if input.chars().all(lowercases_to_self) {
            Ok(input.into())
        } else {
            Ok(Cow::Owned(
                input.chars().flat_map(|c| c.to_lowercase()).collect(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_lowercase_matches_legacy() {
        let n = Lowercase;
        for input in &["HELLO", "Hello World", "abc", "", "ÀÉ", "ΟΔΟΣ"] {
            let mut ns = NormalizedString::from(*input);
            Normalizer::normalize(&n, &mut ns).unwrap(); // legacy oracle
            assert_eq!(
                ns.get(),
                &*pipeline::Normalizer::normalize(&n, input).unwrap()
            );
        }
    }
}
