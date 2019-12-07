use crate::tokenizer::Decoder;

pub struct WordPiece;

impl Decoder for WordPiece {
    fn decode(&self, tokens: Vec<String>) -> String {
        tokens.join(" ").replace(" ##", "")
    }
}
