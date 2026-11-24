use std::borrow::Cow;

use crate::pipeline;
use crate::tokenizer::Result;
use crate::utils::lowercase::{lowercases_to_self, to_lowercase};

/// Lowercases the input
#[derive(Copy, Clone, Debug)]
pub struct Lowercase;

impl pipeline::Normalize for Lowercase {
    fn normalize<'a>(&self, input: &'a str, _offset: usize) -> Result<Cow<'a, str>> {
        if input.chars().all(lowercases_to_self) {
            Ok(input.into())
        } else {
            Ok(Cow::Owned(input.chars().flat_map(to_lowercase).collect()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expected values were captured from the legacy `NormalizedString` normalizer this test used
    /// to compare against, on the commit that removed it -- so they still pin exactly what the two
    /// implementations agreed on, without keeping the legacy code alive to ask.
    #[test]
    fn pipeline_lowercase() {
        let n = Lowercase;
        for (input, expected) in [
            ("HELLO", "hello"),
            ("Hello World", "hello world"),
            ("abc", "abc"),
            ("", ""),
            ("ÀÉ", "àé"),
            ("ΟΔΟΣ", "οδοσ"),
        ] {
            assert_eq!(
                &*pipeline::Normalize::normalize(&n, input, 0).unwrap(),
                expected,
                "input={input:?}"
            );
        }
    }
}
