use crate::tokenizer::{NormalizedString, Normalizer, Result};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize, Serializer};

#[derive(Copy, Clone, Debug, Deserialize)]
pub struct Strip {
    strip_left: bool,
    strip_right: bool,
}

impl Serialize for Strip {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut m = serializer.serialize_struct("Strip", 5)?;
        m.serialize_field("type", "BertNormalizer")?;
        m.serialize_field("strip_left", &self.strip_left)?;
        m.serialize_field("strip_right", &self.strip_right)?;
        m.end()
    }
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
