use crate::tokenizer::{Decoder, Result};

use compact_str::{format_compact, CompactString};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Clone, Debug, Serialize)]
/// The WordPiece decoder takes care of decoding a list of wordpiece tokens
/// back into a readable string.
#[serde(tag = "type")]
#[non_exhaustive]
pub struct WordPiece {
    /// The prefix to be used for continuing subwords
    pub prefix: CompactString,
    /// Whether to cleanup some tokenization artifacts (spaces before punctuation, ...)
    pub cleanup: bool,
}

impl WordPiece {
    pub fn new(prefix: CompactString, cleanup: bool) -> Self {
        Self { prefix, cleanup }
    }
}

impl Default for WordPiece {
    fn default() -> Self {
        Self {
            prefix: "##".into(),
            cleanup: true,
        }
    }
}
pub fn cleanup(dirty_input: &str) -> CompactString {
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
        .into()
}

impl Decoder for WordPiece {
    fn decode_chain(&self, mut tokens: Vec<CompactString>) -> Result<Vec<CompactString>> {
        tokens
            .iter_mut()
            .enumerate()
            .map(|(i, token)| {
                if i != 0 {
                    if token.starts_with(&*self.prefix) {
                        *token = token.replacen(&*self.prefix, "", 1).into();
                    } else {
                        *token = format_compact!(" {token}");
                    }
                }
                if self.cleanup {
                    *token = cleanup(token);
                }
                Ok(token.clone())
            })
            .collect::<Result<_>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wordpiece_decoder() {
        let decoder = WordPiece::new("##".into(), false);

        assert_eq!(
            decoder
                .decode(vec![
                    "##uelo".into(),
                    "Ara".into(),
                    "##új".into(),
                    "##o".into(),
                    "No".into(),
                    "##guera".into()
                ])
                .unwrap(),
            "##uelo Araújo Noguera"
        );
    }
}
