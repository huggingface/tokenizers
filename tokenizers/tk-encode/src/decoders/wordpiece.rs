use std::mem::replace;

use crate::{
    pipeline::{self, DecoderState},
    tokenizer::{Decoder, Result},
};

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

pub fn cleanup(dirty_input: &str) -> String {
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
    fn decode_chain(&self, mut tokens: Vec<String>) -> Result<Vec<String>> {
        for (i, token) in tokens.iter_mut().enumerate() {
            if i != 0 {
                if let Some(tk) = token.strip_prefix(&self.prefix) {
                    *token = tk.to_string();
                } else {
                    *token = format!(" {token}");
                }
            }
            if self.cleanup {
                *token = cleanup(token);
            }
        }
        Ok(tokens)
    }
}

const CLEANUP_LIST: [&[u8]; 10] = [
    b".", b"?", b"!", b",", b"n't", b"'m", b"do not", b"'s", b"'ve", b"'re",
];

impl pipeline::Decoder for WordPiece {
    fn decode_token(
        &self,
        state: &mut DecoderState,
        _token_id: u32,
        token_bytes: &[u8],
        decoded: &mut Vec<u8>,
    ) -> Result<()> {
        if !replace(&mut state.started, true) {
            decoded.extend(token_bytes)
        } else if token_bytes.starts_with(self.prefix.as_bytes()) {
            decoded.extend_from_slice(&token_bytes[self.prefix.len()..]);
        } else if self.cleanup && CLEANUP_LIST.contains(&token_bytes) {
            decoded.extend(token_bytes);
        } else {
            decoded.push(b' ');
            decoded.extend(token_bytes);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wordpiece_decoder() {
        let decoder = WordPiece::new("##".to_string(), false);

        assert_eq!(
            decoder
                .decode(vec![
                    "##uelo".to_string(),
                    "Ara".to_string(),
                    "##új".to_string(),
                    "##o".to_string(),
                    "No".to_string(),
                    "##guera".to_string()
                ])
                .unwrap(),
            "##uelo Araújo Noguera"
        );
    }
}
