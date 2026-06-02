use serde::{Deserialize, Serialize};

/// Progress output format for training operations.
///
/// Controls how progress information is reported during tokenizer training.
/// Default is `Indicatif` which shows interactive terminal progress bars.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ProgressFormat {
    /// Interactive terminal progress bars using indicatif (default behavior)
    #[default]
    Indicatif,
    /// Machine-readable JSON lines to stderr for programmatic consumption
    JsonLines,
    /// No progress output
    Silent,
}

#[cfg(feature = "progressbar")]
pub(crate) use indicatif::{ProgressBar, ProgressStyle};

#[cfg(not(feature = "progressbar"))]
mod progressbar {
    use std::borrow::Cow;
    pub struct ProgressBar;
    impl ProgressBar {
        pub fn new(_length: u64) -> Self {
            Self {}
        }

        pub fn set_length(&self, _length: u64) {}
        pub fn set_message(&self, _message: impl Into<Cow<'static, str>>) {}
        pub fn finish(&self) {}
        pub fn reset(&self) {}
        pub fn inc(&self, _inc: u64) {}
        pub fn set_style(&self, _style: ProgressStyle) {}
    }

    pub struct ProgressStyle {}
    impl ProgressStyle {
        pub fn default_bar() -> Self {
            Self {}
        }
        pub fn template(self, _template: &str) -> Result<Self, String> {
            Ok(self)
        }
    }
}
#[cfg(not(feature = "progressbar"))]
pub(crate) use progressbar::{ProgressBar, ProgressStyle};
