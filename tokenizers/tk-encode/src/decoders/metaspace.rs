//! The `Metaspace` **decoder**.
//! To use in conjunction with the
//! [`MetaspaceNormalizer`]: crate::normalizers::metaspace::MetaspaceNormalizer.

use crate::tokenizer::{Decoder, Result};

/// Enum representing options for the metaspace prepending scheme.
#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum PrependScheme {
    /// Specifies that the scheme should be prepended only once, on the first split.
    First,
    Never,
    Always,
}

impl std::fmt::Display for PrependScheme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::First => "first",
            Self::Never => "never",
            Self::Always => "always",
        })
    }
}

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

    pub fn set_replacement(&mut self, replacement: char) {
        self.replacement = replacement;
    }
}

impl Default for MetaspaceDecoder {
    fn default() -> Self {
        Self::new('▁', PrependScheme::Always)
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
