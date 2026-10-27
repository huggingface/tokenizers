use std::borrow::Cow;

use crate::pipeline;
use crate::tokenizer::Result;

#[derive(Clone, Debug)]
pub struct Prepend {
    pub prepend: String,
    /// When true, leave the input alone if it already starts with [`Self::prepend`].
    ///
    /// Needed for ByteLevel `add_prefix_space`: 0.23 prepends a leading space only when the
    /// string does not already begin with one. Unconditional [`Prepend`] would double-space
    /// those inputs and change token ids.
    pub only_if_absent: bool,
}

impl Prepend {
    pub fn new(prepend: String) -> Self {
        Self {
            prepend,
            only_if_absent: false,
        }
    }

    pub fn only_if_absent(mut self, only_if_absent: bool) -> Self {
        self.only_if_absent = only_if_absent;
        self
    }
}

impl pipeline::Normalizer for Prepend {
    fn normalize<'a>(&self, input: &'a str, _offset: usize) -> Result<Cow<'a, str>> {
        if input.is_empty() || self.prepend.is_empty() {
            return Ok(Cow::Borrowed(input));
        }
        if self.only_if_absent && input.starts_with(&self.prepend) {
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
                &*pipeline::Normalizer::normalize(&n, input, 0).unwrap(),
                expected,
                "input={input:?}"
            );
        }
    }

    #[test]
    fn only_if_absent_skips_when_already_prefixed() {
        let n = Prepend::new(" ".to_string()).only_if_absent(true);
        assert_eq!(
            &*pipeline::Normalizer::normalize(&n, "Hello", 0).unwrap(),
            " Hello"
        );
        assert_eq!(
            &*pipeline::Normalizer::normalize(&n, " Hello", 0).unwrap(),
            " Hello"
        );
        // Unconditional Prepend still doubles.
        let always = Prepend::new(" ".to_string());
        assert_eq!(
            &*pipeline::Normalizer::normalize(&always, " Hello", 0).unwrap(),
            "  Hello"
        );
    }
}
