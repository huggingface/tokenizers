use std::borrow::Cow;

use crate::pipeline;
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

impl pipeline::Normalizer for ByteLevel {
    fn normalize<'a>(&self, input: &'a str) -> Cow<'a, str> {
        let table = &*BYTES_CHAR_LOOKUP;
        let mut out = String::with_capacity(input.len());
        for &b in input.as_bytes() {
            out.push(table[b as usize]);
        }
        Cow::Owned(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_byte_level_matches_legacy() {
        let n = ByteLevel::new();
        for input in &["Hello world", "Hello 我今天", "abc", ""] {
            let mut ns = NormalizedString::from(*input);
            Normalizer::normalize(&n, &mut ns).unwrap(); // legacy oracle
            assert_eq!(ns.get(), &*pipeline::Normalizer::normalize(&n, input));
        }
    }
}
