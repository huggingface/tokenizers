use std::borrow::Cow;

use crate::pipeline;
use crate::tokenizer::Result;

#[derive(Clone, Debug)]
pub struct Prepend {
    pub prepend: String,
}

impl Prepend {
    pub fn new(prepend: String) -> Self {
        Self { prepend }
    }
}

impl pipeline::Normalizer for Prepend {
    fn normalize<'a>(&self, input: &'a str, _is_sequence_start: bool) -> Result<Cow<'a, str>> {
        if input.is_empty() || self.prepend.is_empty() {
            return Ok(Cow::Borrowed(input));
        }
        Ok(Cow::Owned(format!(
            "{prepend}{input}",
            prepend = self.prepend
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expected values were captured from the legacy `NormalizedString` normalizer this test used
    /// to compare against, on the commit that removed it -- so they still pin exactly what the two
    /// implementations agreed on, without keeping the legacy code alive to ask.
    #[test]
    fn pipeline_prepend() {
        let n = Prepend::new("\u{2581}".to_string());
        for (input, expected) in [("Hello", "▁Hello"), ("world", "▁world"), ("", "")] {
            assert_eq!(
                &*pipeline::Normalizer::normalize(&n, input, true).unwrap(),
                expected,
                "input={input:?}"
            );
        }
    }
}
