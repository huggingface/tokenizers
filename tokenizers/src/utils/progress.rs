#[cfg(feature = "progressbar")]
pub(crate) use indicatif::{ProgressBar, ProgressStyle};

#[cfg(not(feature = "progressbar"))]
mod progressbar {
    pub struct ProgressBar;
    impl ProgressBar {
        pub fn new(_length: u64) -> Self {
            Self {}
        }

        pub fn set_length(&self, _length: u64) {}
        pub fn set_draw_delta(&self, _draw_delta: u64) {}
        pub fn set_message(&self, _message: &str) {}
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
        pub fn template(self, _template: &str) -> Self {
            self
        }
    }
}
#[cfg(not(feature = "progressbar"))]
pub(crate) use progressbar::{ProgressBar, ProgressStyle};
