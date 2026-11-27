//! The `Metaspace` **decoder**: it turns back into spaces the delimiters the [`Metaspace`] piece
//! normalizer wrote.
//!
//! [`Metaspace`]: crate::piece_normalizers::Metaspace

use crate::piece_normalizers::metaspace::PrependScheme;
use crate::tokenizer::{Decode, Result};

/// Turns the metaspace replacement character back into a space.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetaspaceDecoder {
    replacement: char,
    pub prepend_scheme: PrependScheme,
}

impl MetaspaceDecoder {
    pub fn new(replacement: char, prepend_scheme: PrependScheme) -> Self {
        Self {
            replacement,
            prepend_scheme,
        }
    }

    pub fn get_replacement(&self) -> char {
        self.replacement
    }
}

impl Default for MetaspaceDecoder {
    fn default() -> Self {
        Self::new('▁', PrependScheme::Always)
    }
}

impl Decode for MetaspaceDecoder {
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
        let decoder = MetaspaceDecoder::new('▁', PrependScheme::Always);
        let res = decoder
            .decode_chain(vec!["▁Hey".into(), "▁friend!".into()])
            .unwrap();
        assert_eq!(res, vec!["Hey", " friend!"]);

        let decoder = MetaspaceDecoder::new('▁', PrependScheme::Never);
        let res = decoder
            .decode_chain(vec!["▁Hey".into(), "▁friend!".into()])
            .unwrap();
        assert_eq!(res, vec![" Hey", " friend!"]);
    }
}
