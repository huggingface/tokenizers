use crate::tokenizer::{Decoder, Result};

use serde::{Deserialize, Serialize};

#[derive(Deserialize, Clone, Debug, Serialize)]
/// Allows decoding Original BPE by joining all the tokens and then replacing
/// the suffix used to identify end-of-words by whitespaces
#[serde(tag = "type")]
#[non_exhaustive]
pub struct BPEDecoder {
    pub suffix: String,
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
