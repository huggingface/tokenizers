use crate::{PaddingParams, TruncationParams};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodeOptions {
    pub add_special_tokens: bool,
    pub padding: Override<PaddingParams>,
    pub truncation: Override<TruncationParams>,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            add_special_tokens: true,
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
