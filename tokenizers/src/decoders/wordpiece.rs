use crate::tokenizer::{Decoder, Result};

pub struct WordPiece {
    prefix: String,
}

impl WordPiece {
    pub fn new(prefix: String) -> Self {
        Self { prefix }
    }
}

impl Default for WordPiece {
    fn default() -> Self {
        Self {
            prefix: String::from("##"),
        }
    }
}

impl Decoder for WordPiece {
    fn decode(&self, tokens: Vec<String>) -> Result<String> {
        Ok(tokens.join(" ").replace(&format!(" {}", self.prefix), ""))
    }
}
