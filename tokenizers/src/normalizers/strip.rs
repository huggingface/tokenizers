use crate::tokenizer::{NormalizedString, Normalizer, Result};

pub struct Strip {
    strip_left: bool,
    strip_right: bool,
}

impl Strip {
    pub fn new(strip_left: bool, strip_right: bool) -> Self {
        Strip {
            strip_left,
            strip_right,
        }
    }
}

impl Normalizer for Strip {
    /// Strip the normalized string inplace
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        if self.strip_left && self.strip_right {
            // Fast path
            normalized.strip();
        } else {
            if self.strip_left {
                normalized.lstrip();
            }

            if self.strip_right {
                normalized.rstrip();
            }
        }

        Ok(())
    }
}
