use crate::tokenizer::{Decoder, Result};

use serde::{Deserialize, Serialize};

#[derive(Deserialize, Clone, Debug, Serialize)]
/// The WordPiece decoder takes care of decoding a list of wordpiece tokens
/// back into a readable string.
#[serde(tag = "type")]
#[non_exhaustive]
pub struct WordPiece {
    /// The prefix to be used for continuing subwords
    pub prefix: String,
    /// Whether to cleanup some tokenization artifacts (spaces before punctuation, ...)
    pub cleanup: bool,
}

impl WordPiece {
    pub fn new(prefix: String, cleanup: bool) -> Self {
        Self { prefix, cleanup }
    }
}

impl Default for WordPiece {
    fn default() -> Self {
        Self {
            prefix: "##".to_owned(),
            cleanup: true,
        }
    }
}
pub fn cleanup(dirty_input: String) -> String {
    dirty_input
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
        .replace(" 're", "'re")
}

impl Decoder for WordPiece {
    fn decode(&self, mut tokens: Vec<String>) -> Result<Vec<String>> {
        tokens
            .iter_mut()
            .enumerate()
            .map(|(i, token)| {
                if token.starts_with(&self.prefix) {
                    *token = token.replacen(&self.prefix, "", 1);
                } else if i != 0 {
                    *token = format!(" {}", token);
                }
                if self.cleanup {
                    *token = cleanup(token.clone());
                }
                Ok(token.to_string())
            })
            .collect::<Result<_>>()
    }
}
