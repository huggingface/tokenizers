use std::borrow::Cow;

use crate::pipeline;
use crate::tokenizer::Result;
use crate::utils::byte_level::BYTES_CHAR_LOOKUP;
use ahash::AHashSet;

#[derive(Clone, Debug)]
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

    /// Expected values were captured from the legacy `NormalizedString` normalizer this test used
    /// to compare against, on the commit that removed it -- so they still pin exactly what the two
    /// implementations agreed on, without keeping the legacy code alive to ask.
    #[test]
    fn pipeline_byte_level() {
        let n = ByteLevel::new();
        for (input, expected) in [
            ("Hello world", "HelloĠworld"),
            ("Hello 我今天", "HelloĠæĪĳä»Ĭå¤©"),
            ("abc", "abc"),
            ("", ""),
        ] {
            assert_eq!(
                &*pipeline::Normalizer::normalize(&n, input).unwrap(),
                expected,
                "input={input:?}"
            );
        }
    }
}
