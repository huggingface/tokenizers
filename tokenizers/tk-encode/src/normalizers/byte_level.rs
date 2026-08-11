use std::borrow::Cow;

use crate::pipeline;
use crate::tokenizer::Result;
use crate::utils::byte_level::BYTES_CHAR_LOOKUP;
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

impl pipeline::Normalizer for ByteLevel {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
        let table = &*BYTES_CHAR_LOOKUP;
        let mut out = String::with_capacity(2 * input.len());
        for &b in input.as_bytes() {
            out.push(table[b as usize]);
        }
        Ok(Cow::Owned(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalizers::assert_normalizes;

    #[test]
    fn every_byte_becomes_one_printable_char() {
        assert_normalizes(
            &ByteLevel::new(),
            &[
                ("Hello world", "HelloĠworld"),
                // A multi-byte codepoint turns into one char per byte
                ("Hello 我今天", "HelloĠæĪĳä»Ĭå¤©"),
                (
                    "Hello 我今天能为你做什么",
                    "HelloĠæĪĳä»Ĭå¤©èĥ½ä¸ºä½łåģļä»Ģä¹Ī",
                ),
                ("abc", "abc"),
                ("", ""),
            ],
        );
    }
}
