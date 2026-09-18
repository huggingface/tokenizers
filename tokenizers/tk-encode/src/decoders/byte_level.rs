//! The `ByteLevel` **decoder**.
//! It just takes the printable character and converts it to the real character. This used to be
//! the default as we wanted to make sure serialized vocabulary was "readable" as such non
//! printable ascii were replace with the next printable.
//!
//! This is the whole reason you see 'Ġ' everywhere! It's the encode " " char.

use crate::tokenizer::{Decoder, Result};
use crate::utils::byte_level::char_to_byte;

/// Maps the byte-level alphabet back to the bytes it stands for.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ByteLevelDecoder {}

impl ByteLevelDecoder {
    pub fn new() -> Self {
        Self {}
    }
}

impl Decoder for ByteLevelDecoder {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        let toks = tokens
            .into_iter()
            .flat_map(|t| {
                t.chars()
                    .try_fold(vec![], |mut acc, c| {
                        char_to_byte(c).map(|b| {
                            acc.push(b);
                            acc
                        })
                    })
                    .unwrap_or_else(|| t.as_bytes().to_vec())
            })
            .collect::<Vec<u8>>();
        Ok(vec![String::from_utf8_lossy(&toks).to_string()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoding() {
        let bytelevel = ByteLevelDecoder::default();
        assert_eq!(
            bytelevel
                .decode_chain(
                    vec![
                        "Hello", "Ġmy", "Ġfriend", ",", "Ġhow", "Ġis", "Ġyour", "Ġday", "Ġgoing",
                        "?"
                    ]
                    .into_iter()
                    .map(|s| s.into())
                    .collect::<Vec<String>>()
                )
                .unwrap(),
            vec!["Hello my friend, how is your day going?"]
        );
    }

    /// A token outside the byte-level alphabet is passed through as its own bytes rather than
    /// dropped -- which is what keeps an added token like `[PA D]` readable.
    #[test]
    fn decode_unknown_characters() {
        let byte_level = ByteLevelDecoder::default();
        assert_eq!(
            byte_level
                .decode_chain(vec![
                    "Hello".into(),
                    "Ġthere".into(),
                    "Ġdear".into(),
                    "Ġfriend!".into(),
                    "Ġ".into(),
                    "[PA D]".into()
                ])
                .unwrap(),
            vec!["Hello there dear friend! [PA D]"]
        );
    }
}
