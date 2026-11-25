use std::num::NonZeroUsize;

use crate::{PaddingParams, TruncationParams};

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct EncodeOptions {
    /// Whether the post-processor should add special tokens to the sequence,
    /// e.g. `[CLS]`, `[SEP]`, `<|endoftext|>`, `</s>`. Defaults to `true`.
    pub add_special_tokens: bool,
    /// Whether special tokens found in the sequence should go through the tokenizer model (`true`)
    /// or be replaced by their id in the vocabulary (`false`). Defaults to `false`.
    pub encode_special_tokens: bool,
    /// Override the tokenizer's padding options. Defaults to [`Override::InheritConfig`].
    pub padding: Override<PaddingParams>,
    pub truncation: Override<TruncationParams>,
    /// The most threads this call uses, the calling thread included, on top of
    /// [`max_threads`](crate::parallelism::max_threads). `Some(1)` encodes serially. Defaults to
    /// `None`, no cap.
    pub max_threads: Option<NonZeroUsize>,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            add_special_tokens: true,
            encode_special_tokens: false,
            padding: Override::InheritConfig,
            truncation: Override::InheritConfig,
            max_threads: None,
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
