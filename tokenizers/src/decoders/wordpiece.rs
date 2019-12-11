use crate::tokenizer::{Decoder, Result};

pub struct WordPiece;

impl Decoder for WordPiece {
    fn decode(&self, tokens: Vec<String>) -> Result<String> {
        Ok(tokens.join(" ").replace(" ##", ""))
    }
}
