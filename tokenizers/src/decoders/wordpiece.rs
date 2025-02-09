use crate::tokenizer::{Decoder, Result};

use compact_str::{format_compact, CompactString, ToCompactString};
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
pub fn cleanup(dirty_input: impl ToCompactString) -> CompactString {
    dirty_input
        .to_compact_string()
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
    fn decode_chain<T: ToCompactString>(
        &self,
        tokens: Vec<T>,
    ) -> Result<Vec<CompactString>> {
        tokens
            .into_iter()
            .map(|t| t.to_compact_string())
            .enumerate()
            .map(|(i, mut token)| {
                if i != 0 {
                    if token.starts_with(&*self.prefix) {
                        token = token.replacen(&*self.prefix, "", 1).to_compact_string();
                    } else {
                        token = format_compact!(" {}", token);
                    }
                }
                if self.cleanup {
                    token = cleanup(token);
                }
                Ok(token.clone().into())
            })
            .collect::<Result<Vec<CompactString>>>()
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
                    "##uelo".to_owned(),
                    "Ara".to_owned(),
                    "##új".to_owned(),
                    "##o".to_owned(),
                    "No".to_owned(),
                    "##guera".to_owned()
                ])
                .unwrap(),
            "##uelo Araújo Noguera"
        );
    }
}
