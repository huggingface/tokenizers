use serde::{Deserialize, Serialize};

use crate::normalizers::NormalizerWrapper;
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

    pub fn get_normalizers(&self) -> &[NormalizerWrapper] {
        &self.normalizers
    }

    pub fn get_normalizers_mut(&mut self) -> &mut [NormalizerWrapper] {
        &mut self.normalizers
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
