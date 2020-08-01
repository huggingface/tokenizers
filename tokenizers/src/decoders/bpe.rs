use crate::tokenizer::{Decoder, Result};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize, Serializer};

#[derive(Deserialize, Clone, Debug)]
/// Allows decoding Original BPE by joining all the tokens and then replacing
/// the suffix used to identify end-of-words by whitespaces
pub struct BPEDecoder {
    suffix: String,
}

impl BPEDecoder {
    pub fn new(suffix: String) -> Self {
        BPEDecoder { suffix }
    }
}

impl Default for BPEDecoder {
    fn default() -> Self {
        BPEDecoder::new("</w>".into())
    }
}

impl Decoder for BPEDecoder {
    fn decode(&self, tokens: Vec<String>) -> Result<String> {
        Ok(tokens.join("").replace(&self.suffix, " ").trim().to_owned())
    }
}

impl Serialize for BPEDecoder {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut m = serializer.serialize_struct("BPEDecoder", 2)?;
        m.serialize_field("type", "BPEDecoder")?;
        m.serialize_field("suffix", &self.suffix)?;
        m.end()
    }
}
