use std::borrow::Cow;

use crate::pipeline;
use crate::tokenizer::{Normalizer, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
pub struct Prepend {
    pub prepend: String,
}

impl Prepend {
    pub fn new(prepend: String) -> Self {
        Self { prepend }
    }
}

impl Normalizer for Prepend {}

impl pipeline::Normalizer for Prepend {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
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
    use crate::normalizers::assert_normalizes;

    #[test]
    fn prepend_puts_its_string_in_front() {
        assert_normalizes(
            &Prepend::new("▁".to_string()),
            &[
                ("Hello", "▁Hello"),
                ("world", "▁world"),
                // Nothing to prepend to
                ("", ""),
            ],
        );
    }
}
