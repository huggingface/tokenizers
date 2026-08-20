use crate::tokenizer::{Decoder, Result};

#[derive(Clone, Debug)]
/// Allows decoding Original BPE by joining all the tokens and then replacing
/// the suffix used to identify end-of-words by whitespaces
///
/// The tag is spelled `BPEDecoder`, not `BPE` -- that is what `#[serde(tag = "type")]` on a struct
/// of this name writes, and what `DecoderWrapper`'s `EnumType` matches. The tag is *not* required on
/// the way in: a bare `#[serde(tag = ...)]` only adds it on the way out, and what rejects a tag-less
/// object here is the missing `suffix` field.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "type"))]
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
