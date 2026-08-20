//! The `Sequence` normalizer. Lives here rather than in `tk-encode` because it is a
//! `Vec<NormalizerWrapper>`, and a `Vec` of a type can only be parameterised where that type is.

use std::borrow::Cow;

use serde::{Deserialize, Serialize};

use tk_encode::pipeline;
use tk_encode::{NormalizedString, Normalizer, Result};

use crate::normalizers::NormalizerWrapper;

#[derive(Clone, Debug, Deserialize, Serialize)]
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
