use crate::tokenizer::{NormalizedString, Normalizer, Result};

/// Allows concatenating multiple other Normalizer as a Sequence.
/// All the normalizers run in sequence in the given order against the same NormalizedString.
pub struct Sequence {
    normalizers: Vec<Box<dyn Normalizer + Sync>>,
}

impl Sequence {
    pub fn new(normalizers: Vec<Box<dyn Normalizer + Sync>>) -> Self {
        Self { normalizers }
    }
}

impl Normalizer for Sequence {
    fn normalize(&self, mut normalized: &mut NormalizedString) -> Result<()> {
        for normalizer in &self.normalizers {
            normalizer.normalize(&mut normalized)?;
        }
        Ok(())
    }
}

/// Lowercases the input
pub struct Lowercase;
impl Normalizer for Lowercase {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.lowercase();
        Ok(())
    }
}
