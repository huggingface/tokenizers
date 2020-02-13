use crate::tokenizer::{Decoder, Result};

/// The WordPiece decoder takes care of decoding a list of wordpiece tokens
/// back into a readable string.
pub struct WordPiece {
    /// The prefix to be used for continuing subwords
    prefix: String,
    /// Whether to cleanup some tokenization artifacts (spaces before punctuation, ...)
    cleanup: bool,
}

impl WordPiece {
    pub fn new(prefix: String, cleanup: bool) -> Self {
        Self { prefix, cleanup }
    }
}

impl Default for WordPiece {
    fn default() -> Self {
        Self {
            prefix: String::from("##"),
            cleanup: true,
        }
    }
}

impl Decoder for WordPiece {
    fn decode(&self, tokens: Vec<String>) -> Result<String> {
        let mut output = tokens.join(" ").replace(&format!(" {}", self.prefix), "");
        if self.cleanup {
            output = output
                .replace(" .", ".")
                .replace(" ?", "?")
                .replace(" !", "!")
                .replace(" ,", ",")
                .replace(" ' ", "'")
                .replace(" n't", "n't")
                .replace(" 'm", "'m")
                .replace(" do not", " don't")
                .replace(" 's", "'s")
                .replace(" 've", "'ve")
                .replace(" 're", "'re");
        }

        Ok(output)
    }
}
