//! The `Metaspace` **decoder**.
//!
//! A different type from the `Metaspace` *pre-tokenizer*, deliberately — see
//! [`crate::decoders::replace`] for why.
//!
//! `PrependScheme` is shared, and that is fine: it is plain data describing a setting, not a
//! component playing a role. What was worth separating is the thing that sits in a wrapper.

use crate::tokenizer::{Decoder, Result};

pub use crate::pre_tokenizers::metaspace::PrependScheme;

/// Turns the metaspace replacement character back into a space.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetaspaceDecoder {
    replacement: char,
    pub prepend_scheme: PrependScheme,
    /// Carried for round-tripping only; decoding does not read it.
    pub split: bool,
}

impl MetaspaceDecoder {
    pub fn new(replacement: char, prepend_scheme: PrependScheme, split: bool) -> Self {
        Self {
            replacement,
            prepend_scheme,
            split,
        }
    }

    pub fn get_replacement(&self) -> char {
        self.replacement
    }

    pub fn set_replacement(&mut self, replacement: char) {
        self.replacement = replacement;
    }
}

impl Default for MetaspaceDecoder {
    fn default() -> Self {
        Self::new('▁', PrependScheme::Always, true)
    }
}

impl Decoder for MetaspaceDecoder {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        Ok(tokens
            .iter()
            .enumerate()
            .map(|(i, token)| {
                token
                    .chars()
                    .flat_map(|c| {
                        if c == self.replacement {
                            if i == 0 && self.prepend_scheme != PrependScheme::Never {
                                None
                            } else {
                                Some(' ')
                            }
                        } else {
                            Some(c)
                        }
                    })
                    .collect::<String>()
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode() {
        let decoder = MetaspaceDecoder::new('▁', PrependScheme::Always, true);
        let res = decoder
            .decode_chain(vec!["▁Hey".into(), "▁friend!".into()])
            .unwrap();
        assert_eq!(res, vec!["Hey", " friend!"]);

        let decoder = MetaspaceDecoder::new('▁', PrependScheme::Never, true);
        let res = decoder
            .decode_chain(vec!["▁Hey".into(), "▁friend!".into()])
            .unwrap();
        assert_eq!(res, vec![" Hey", " friend!"]);
    }
}
