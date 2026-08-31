use std::borrow::Cow;

use crate::pipeline;
use crate::tokenizer::Result;

// The `Sequence` normalizer is a `Vec<NormalizerWrapper>`, so it lives in `tk-convert` with
// the wrapper it is parameterised by.

/// Lowercases the input
#[derive(Copy, Clone, Debug)]
pub struct Lowercase;

/// Whether lowercasing `c` leaves it unchanged (a single, identical char)
pub(crate) fn lowercases_to_self(c: char) -> bool {
    let mut it = c.to_lowercase();
    matches!((it.next(), it.next()), (Some(first), None) if first == c)
}

impl pipeline::Normalizer for Lowercase {
    fn normalize<'a>(&self, input: &'a str, _is_first_chunk: bool) -> Result<Cow<'a, str>> {
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
                &*pipeline::Normalizer::normalize(&n, input, true).unwrap(),
                expected,
                "input={input:?}"
            );
        }
    }
}
