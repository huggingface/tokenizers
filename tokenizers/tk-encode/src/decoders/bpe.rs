use crate::tokenizer::{Decoder, Result};

#[derive(Clone, Debug)]
/// Allows decoding Original BPE by joining all the tokens and then replacing
/// the suffix used to identify end-of-words by whitespaces
///
/// The `"type": "BPEDecoder"` envelope this used to derive lives in `tk-convert`'s
/// `decoders::mirror::bpe`.
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
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        let n = tokens.len() - 1;
        Ok(tokens
            .into_iter()
            .enumerate()
            .map(|(i, token)| {
                let replacement = if i == n { "" } else { " " };
                token.replace(&self.suffix, replacement)
            })
            .collect())
    }
}
