use crate::tokenizer::{Decoder, Result};

use serde::{Deserialize, Serialize};

#[derive(Deserialize, Clone, Debug, Serialize, Default)]
/// Strip is a simple trick which converts tokens looking like `<0x61>`
/// to pure bytes, and attempts to make them into a string. If the tokens
/// cannot be decoded you will get � instead for each inconvertable byte token
#[serde(tag = "type")]
#[non_exhaustive]
pub struct Strip {
    pub left: usize,
    pub right: usize,
}

impl Strip {
    pub fn new(left: usize, right: usize) -> Self {
        Self { left, right }
    }
}

impl Decoder for Strip {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        Ok(tokens
            .into_iter()
            .map(|token| token[self.left..token.len() - self.right].to_string())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode() {
        let decoder = Strip::new(1, 0);
        let res = decoder
            .decode_chain(vec!["Hey".into(), " friend!".into()])
            .unwrap();
        assert_eq!(res, vec!["ey", "friend!"]);

        let decoder = Strip::new(0, 1);
        let res = decoder
            .decode_chain(vec!["Hey".into(), " friend!".into()])
            .unwrap();
        assert_eq!(res, vec!["He", " friend"]);
    }
}
