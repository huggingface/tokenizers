use std::borrow::Cow;

use serde::{Deserialize, Serialize};

use crate::normalizers::NormalizerWrapper;
use crate::pipeline;
use crate::tokenizer::{NormalizedString, Normalizer, Result};
use crate::utils::macro_rules_attribute;

#[derive(Clone, Deserialize, Debug, Serialize)]
#[serde(tag = "type")]
/// Allows concatenating multiple other Normalizer as a Sequence.
/// All the normalizers run in sequence in the given order against the same NormalizedString.
pub struct Sequence {
    normalizers: Vec<NormalizerWrapper>,
}

impl Sequence {
    pub fn new(normalizers: Vec<NormalizerWrapper>) -> Self {
        Self { normalizers }
    }
}

impl AsRef<[NormalizerWrapper]> for Sequence {
    fn as_ref(&self) -> &[NormalizerWrapper] {
        &self.normalizers
    }
}

impl AsMut<[NormalizerWrapper]> for Sequence {
    fn as_mut(&mut self) -> &mut [NormalizerWrapper] {
        &mut self.normalizers
    }
}

impl IntoIterator for Sequence {
    type Item = NormalizerWrapper;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.normalizers.into_iter()
    }
}

impl Normalizer for Sequence {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        for normalizer in &self.normalizers {
            normalizer.normalize(normalized)?;
        }
        Ok(())
    }
}

impl pipeline::Normalizer for Sequence {
    fn normalize<'a>(&self, input: &'a str) -> Cow<'a, str> {
        let mut cow: Cow<'a, str> = Cow::Borrowed(input);
        for normalizer in &self.normalizers {
            cow = match cow {
                // Still borrowing `input` ('a): chain directly so an all-no-op
                // sequence stays zero-alloc and returns a borrow of `input`.
                Cow::Borrowed(s) => pipeline::Normalizer::normalize(normalizer, s),
                // Owned locally: the next step may borrow from it, so materialize
                // its result before the local `String` is dropped.
                Cow::Owned(s) => match pipeline::Normalizer::normalize(normalizer, &s) {
                    Cow::Borrowed(b) => Cow::Owned(b.to_owned()),
                    Cow::Owned(o) => Cow::Owned(o),
                },
            };
        }
        cow
    }
}

/// Lowercases the input
#[derive(Copy, Clone, Debug)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct Lowercase;
impl Normalizer for Lowercase {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.lowercase();
        Ok(())
    }
}

impl pipeline::Normalizer for Lowercase {
    fn normalize<'a>(&self, input: &'a str) -> Cow<'a, str> {
        Cow::Owned(input.to_lowercase())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalizers::{StripAccents, NFD};

    #[test]
    fn pipeline_lowercase_matches_legacy() {
        let n = Lowercase;
        for input in &["HELLO", "Hello World", "abc", "", "ÀÉ"] {
            let mut ns = NormalizedString::from(*input);
            Normalizer::normalize(&n, &mut ns).unwrap(); // legacy oracle
            assert_eq!(ns.get(), &*pipeline::Normalizer::normalize(&n, input));
        }
    }

    #[test]
    fn pipeline_sequence_matches_legacy() {
        let n = Sequence::new(vec![NFD.into(), StripAccents.into(), Lowercase.into()]);
        for input in &["Café", "HÉLLO", "abc", ""] {
            let mut ns = NormalizedString::from(*input);
            Normalizer::normalize(&n, &mut ns).unwrap(); // legacy oracle
            assert_eq!(ns.get(), &*pipeline::Normalizer::normalize(&n, input));
        }
    }
}
