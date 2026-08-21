//! The `ByteLevel` **decoder**.
//!
//! A different type from the `ByteLevel` *pre-tokenizer*, deliberately — see
//! [`crate::decoders::replace`] for why. `decoders/mod.rs` used to `pub use` the pre-tokenizer
//! "as a decoder", which is how one type came to sit in three wrappers at once.
//!
//! Decoding needs none of the pre-tokenizer's flags: it is a fixed inverse of the byte→char map.
//! `add_prefix_space`, `trim_offsets` and `use_regex` used to be carried here so a config could be
//! read and written back unchanged; nothing read them, so they are gone and the decoder is now
//! field-less. The pre-tokenizer keeps its own copies, where they are functional.

use crate::tokenizer::{Decoder, Result};
use crate::utils::byte_level::CHAR_BYTES_LOOKUP;

/// Maps the byte-level alphabet back to the bytes it stands for.
///
/// The tag on disk is `ByteLevel`, not the type name: the decoder used to *be* the pre-tokenizer
/// type, and splitting them into two types must not change what a `tokenizer.json` says. Being
/// field-less, its serde lives in `super::serialization` -- the tag has to be *required*, or an
/// untagged `DecoderWrapper` would let this variant claim any object at all.
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
