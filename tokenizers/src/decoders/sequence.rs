use crate::decoders::DecoderWrapper;
use crate::tokenizer::{Decoder, Result};
use crate::utils::macro_rules_attribute;
use pyo3_special_method_derive::{Dict, Dir, Getattr, Repr, Str};
use serde::{Deserialize, Serialize};

#[macro_rules_attribute(impl_serde_type!)]
#[derive(Clone, Debug, Display)]
#[display(
    fmt = "Sequence([{}])",
    "decoders.iter().map(|d| d.to_string()).fold( String::new(), |mut acc, s|{
        if !acc.is_empty(){
            acc.push_str(\", \");
        }
        acc.push_str(&s);
        acc
    })"
)]
pub struct Sequence {
    decoders: Vec<DecoderWrapper>,
}

impl Sequence {
    pub fn new(decoders: Vec<DecoderWrapper>) -> Self {
        Self { decoders }
    }

    pub fn get_decoders(&self) -> &[DecoderWrapper] {
        &self.decoders
    }

    pub fn get_decoders_mut(&mut self) -> &mut [DecoderWrapper] {
        &mut self.decoders
    }
}

impl Decoder for Sequence {
    fn decode_chain(&self, mut tokens: Vec<String>) -> Result<Vec<String>> {
        for decoder in &self.decoders {
            tokens = decoder.decode_chain(tokens)?;
        }
        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoders::ctc::CTC;
    use crate::pre_tokenizers::metaspace::Metaspace;

    #[test]
    fn sequence_basic() {
        let decoders = vec![
            DecoderWrapper::CTC(CTC::default()),
            DecoderWrapper::Metaspace(Metaspace::default()),
        ];
        let decoder = Sequence::new(decoders);
        let tokens: Vec<String> = vec!["▁", "▁", "H", "H", "i", "i", "▁", "y", "o", "u"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let out_tokens = decoder.decode(tokens).unwrap();
        assert_eq!(out_tokens, "Hi you");
    }
}
