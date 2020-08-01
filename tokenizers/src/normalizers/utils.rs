use crate::normalizers::NormalizerWrapper;
use crate::tokenizer::{NormalizedString, Normalizer, Result};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize, Serializer};

#[derive(Clone, Deserialize, Debug)]
/// Allows concatenating multiple other Normalizer as a Sequence.
/// All the normalizers run in sequence in the given order against the same NormalizedString.
pub struct Sequence {
    normalizers: Vec<NormalizerWrapper>,
}

impl Serialize for Sequence {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut m = serializer.serialize_struct("Sequence", 2)?;
        m.serialize_field("type", "Sequence")?;
        m.serialize_field("normalizers", &self.normalizers)?;
        m.end()
    }
}

impl Sequence {
    pub fn new(normalizers: Vec<NormalizerWrapper>) -> Self {
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
#[derive(Copy, Clone, Debug)]
pub struct Lowercase;
impl Normalizer for Lowercase {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.lowercase();
        Ok(())
    }
}

impl_serde_unit_struct!(LowercaseVisitor, Lowercase);
