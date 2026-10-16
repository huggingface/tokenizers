use crate::{PaddingParams, TruncationParams};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodeOptions {
    /// Whether the post-processor should add special tokens to the sequence,
    /// e.g. `[CLS]`, `[SEP]`, `<|endoftext|>`, `</s>`. Defaults to `true`.
    pub add_special_tokens: bool,
    /// Whether special tokens found in the sequence should go through the tokenizer model (`true`)
    /// or be replaced by their id in the vocabulary (`false`). Defaults to `false`.
    pub encode_special_tokens: bool,
    /// Preserve the token strings, including unknown pieces and matched added-token whitespace.
    /// Defaults to `false` so callers that only need ids do not allocate token strings.
    pub return_tokens: bool,
    /// Override the tokenizer's padding options. Defaults to [`Override::InheritConfig`].
    pub padding: Override<PaddingParams>,
    pub truncation: Override<TruncationParams>,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            add_special_tokens: true,
            encode_special_tokens: false,
            return_tokens: false,
            padding: Override::InheritConfig,
            truncation: Override::InheritConfig,
        }
    }
}

impl EncodeOptions {
    pub fn no_specials() -> Self {
        Self {
            add_special_tokens: false,
            ..Default::default()
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub enum Override<T> {
    #[default]
    InheritConfig,
    Off,
    With(T),
}
