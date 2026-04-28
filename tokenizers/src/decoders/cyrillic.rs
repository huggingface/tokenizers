use crate::tokenizer::{Decoder, Result};
use serde::{Deserialize, Serialize};
use regex::Regex;

use crate::normalizers::cyrillic::latin_to_cyrillic;

#[derive(Deserialize, Clone, Debug, Serialize)]
#[non_exhaustive]
/// Decodes Latin text inside <cyr> tags back into Cyrillic,
/// then removes the tags.
pub struct CyrillicDecoder;

impl CyrillicDecoder {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for CyrillicDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder for CyrillicDecoder {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        let re = Regex::new(r"<cyr>(.*?)</cyr>").unwrap();

        Ok(tokens
            .into_iter()
            .map(|token| {
                re.replace_all(&token, |caps: &regex::Captures| {
                    let inner = &caps[1];
                    latin_to_cyrillic(inner)
                })
                .to_string()
            })
            .collect())
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cyrillic_decoder_replaces_and_strips_tags() {
        let decoder = CyrillicDecoder::default();

        let original = "1. Hej<cyr> žabo, Pozdravljam te Ya,</cyr> dobri svete, iz<cyr> godine 2029-te.</cyr>";
        let expected = "1. Hej жабо, Поздрављам те Я, dobri svete, iz године 2029-те.";
        let tokens = vec![original.to_string()];
        let decoded = decoder.decode_chain(tokens).unwrap();

        assert_eq!(decoded[0], expected);
    }
}
