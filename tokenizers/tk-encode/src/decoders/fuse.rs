use crate::tokenizer::{Decoder, Result};
use monostate::MustBe;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
/// Fuse simply fuses all tokens into one big string.
/// It's usually the last decoding step anyway, but this
/// decoder exists incase some decoders need to happen after that
/// step
#[non_exhaustive]
pub struct Fuse {
    #[serde(rename = "type")]
    type_: MustBe!("Fuse"),
}

impl Fuse {
    pub fn new() -> Self {
        Self {
            type_: MustBe!("Fuse"),
        }
    }
}

impl Decoder for Fuse {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        let new_string = tokens.join("");
        Ok(vec![new_string])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode() {
        let decoder = Fuse::new();
        let res = decoder
            .decode_chain(vec!["Hey".into(), " friend!".into()])
            .unwrap();
        assert_eq!(res, vec!["Hey friend!"]);
    }
}
