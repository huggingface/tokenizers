use std::borrow::Cow;

use serde::{Deserialize, Serialize};

use crate::normalizers::NormalizerWrapper;
use crate::pipeline;
use crate::tokenizer::Result;
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

impl pipeline::Normalizer for Sequence {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
        pipeline::normalize_all(&self.normalizers, input)
    }
}

/// Lowercases the input
#[derive(Copy, Clone, Debug)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct Lowercase;

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
    use crate::normalizers::{NFD, Strip, StripAccents, assert_normalizes};

    #[test]
    fn lowercase_folds_case() {
        assert_normalizes(
            &Lowercase,
            &[
                ("HELLO", "hello"),
                ("Hello World", "hello world"),
                ("ÀÉ", "àé"),
                ("ΟΔΟΣ", "οδοσ"),
                ("abc", "abc"),
                ("", ""),
            ],
        );
    }

    #[test]
    fn sequence_runs_each_normalizer_in_order() {
        let n = Sequence::new(vec![NFD.into(), StripAccents.into(), Lowercase.into()]);
        assert_normalizes(
            &n,
            &[
                ("Café", "cafe"),
                ("HÉLLO", "hello"),
                ("ΟΔΟΣ", "οδοσ"),
                ("abc", "abc"),
                ("", ""),
            ],
        );
    }

    #[test]
    fn sequence_keeps_a_borrow_of_an_owned_step() {
        // `Strip` hands back a slice of what `Lowercase` produced. That slice borrows a
        // `String` local to the chain, so it has to be copied out before the `String` is
        // dropped, and it must not be read as "nothing changed".
        let n = Sequence::new(vec![Lowercase.into(), Strip::new(true, true).into()]);
        assert_normalizes(
            &n,
            &[
                ("  HELLO  ", "hello"),
                ("\tMiXeD Case\n", "mixed case"),
                ("NOPAD", "nopad"),
                ("  hello  ", "hello"),
                ("   ", ""),
                ("", ""),
            ],
        );
    }
}
