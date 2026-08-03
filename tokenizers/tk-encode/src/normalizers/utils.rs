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
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
        pipeline::normalize_all(&self.normalizers, input)
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
    use crate::normalizers::{NFD, Strip, StripAccents};

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

    #[test]
    fn pipeline_sequence_matches_legacy() {
        let n = Sequence::new(vec![NFD.into(), StripAccents.into(), Lowercase.into()]);
        for input in &["Café", "HÉLLO", "abc", "", "ΟΔΟΣ"] {
            let mut ns = NormalizedString::from(*input);
            Normalizer::normalize(&n, &mut ns).unwrap(); // legacy oracle
            assert_eq!(
                ns.get(),
                &*pipeline::Normalizer::normalize(&n, input).unwrap()
            );
        }
    }

    #[test]
    fn pipeline_sequence_strip_after_owned_matches_legacy() {
        // Strip returns a sub-borrow of the previous step's owned output —
        // the one case where a Borrowed result must not be mistaken for a no-op
        let n = Sequence::new(vec![Lowercase.into(), Strip::new(true, true).into()]);
        for input in &[
            "  HELLO  ",
            "\tMiXeD Case\n",
            "NOPAD",
            "  hello  ",
            "",
            "   ",
        ] {
            let mut ns = NormalizedString::from(*input);
            Normalizer::normalize(&n, &mut ns).unwrap(); // legacy oracle
            assert_eq!(
                ns.get(),
                &*pipeline::Normalizer::normalize(&n, input).unwrap(),
                "input={input:?}"
            );
        }
    }
}
