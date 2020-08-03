use crate::tokenizer::{Decoder, Result};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize, Serializer};

#[derive(Deserialize, Clone, Debug)]
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

impl Serialize for WordPiece {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut m = serializer.serialize_struct("BPEDecoder", 3)?;
        m.serialize_field("type", "BPEDecoder")?;
        m.serialize_field("prefix", &self.prefix)?;
        m.serialize_field("cleanup", &self.cleanup)?;
        m.end()
    }
}
