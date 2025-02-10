use crate::tokenizer::{Decoder, Result};

use compact_str::{CompactString, ToCompactString};
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
        Self { suffix }
    }
}

impl Default for BPEDecoder {
    fn default() -> Self {
        Self::new("</w>".into())
    }
}

impl Decoder for BPEDecoder {
    fn decode_chain<T: ToCompactString>(
        &self,
        tokens: Vec<T>,
    ) -> Result<Vec<impl ToCompactString>> {
        let n = tokens.len() - 1;
        Ok(tokens
            .into_iter()
            .enumerate()
            .map(|(i, token)| {
                let replacement = if i == n { "" } else { " " };
                token
                    .to_compact_string()
                    .replace(&self.suffix, replacement)
                    .to_compact_string()
            })
            .collect::<Vec<CompactString>>())
    }
}
