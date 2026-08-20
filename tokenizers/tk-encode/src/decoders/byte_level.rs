//! The `ByteLevel` **decoder**.
//!
//! A different type from the `ByteLevel` *pre-tokenizer*, deliberately — see
//! [`crate::decoders::replace`] for why. `decoders/mod.rs` used to `pub use` the pre-tokenizer
//! "as a decoder", which is how one type came to sit in three wrappers at once.
//!
//! Decoding needs none of the three flags: it is a fixed inverse of the byte→char map. They are
//! carried anyway so that reading a config and writing it back is lossless.

use crate::tokenizer::{Decoder, Result};
use crate::utils::byte_level::CHAR_BYTES_LOOKUP;

#[cfg(feature = "serde")]
fn default_true() -> bool {
    true
}

/// Maps the byte-level alphabet back to the bytes it stands for.
///
/// `rename` because the tag on disk is `ByteLevel`: the decoder used to *be* the pre-tokenizer type,
/// and splitting them into two types must not change what a `tokenizer.json` says.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "type", rename = "ByteLevel"))]
pub struct ByteLevelDecoder {
    /// Carried for round-tripping only; decoding does not read it.
    pub add_prefix_space: bool,
    /// Carried for round-tripping only; decoding does not read it.
    pub trim_offsets: bool,
    /// Carried for round-tripping only; decoding does not read it.
    ///
    /// The one decoder field with a serde default, and it is `true`: configs written before
    /// `use_regex` existed have to keep loading with the regex on.
    #[cfg_attr(feature = "serde", serde(default = "default_true"))]
    pub use_regex: bool,
}

impl ByteLevelDecoder {
    pub fn new(add_prefix_space: bool, trim_offsets: bool, use_regex: bool) -> Self {
        Self {
            add_prefix_space,
            trim_offsets,
            use_regex,
        }
    }
}

impl Default for ByteLevelDecoder {
    fn default() -> Self {
        Self::new(true, true, true)
    }
}

impl Decoder for ByteLevelDecoder {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        let toks = tokens
            .into_iter()
            .flat_map(|t| {
                t.chars()
                    .try_fold(vec![], |mut acc, c| {
                        CHAR_BYTES_LOOKUP.get(&c).map(|b| {
                            acc.push(*b);
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
