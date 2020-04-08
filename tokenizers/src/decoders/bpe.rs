use crate::tokenizer::{Decoder, Result};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
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

#[typetag::serde]
impl Decoder for BPEDecoder {
    fn decode(&self, tokens: Vec<String>) -> Result<String> {
        Ok(tokens.join("").replace(&self.suffix, " ").trim().to_owned())
    }
}
