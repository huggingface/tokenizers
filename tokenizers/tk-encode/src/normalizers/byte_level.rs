use crate::tokenizer::{NormalizedString, Normalizer, Result};
use crate::utils::byte_level::{byte_level_transform, BYTES_CHAR_LOOKUP};
use crate::utils::macro_rules_attribute;
use ahash::AHashSet;

#[derive(Clone, Debug)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct ByteLevel;

impl Default for ByteLevel {
    fn default() -> Self {
        Self::new()
    }
}

impl ByteLevel {
    pub fn new() -> Self {
        Self {}
    }

    pub fn alphabet() -> AHashSet<char> {
        BYTES_CHAR_LOOKUP.iter().copied().collect()
    }
}

impl Normalizer for ByteLevel {
    /// Strip the normalized string inplace
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        if !normalized.is_empty() {
            let s = normalized.get();
            normalized.transform(byte_level_transform(s), 0);
        }
        Ok(())
    }
}
