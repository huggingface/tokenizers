//! The `Metaspace` **decoder**.
//! To use in conjunction with the
//! [`MetaspaceNormalizer`]: crate::normalizers::metaspace::MetaspaceNormalizer.

use crate::normalizers::metaspace::PrependScheme;
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
                    .enumerate()
                    .filter_map(|(j, c)| {
                        if c == self.replacement {
                            // Only strip the single replacement char that was
                            // prepended during pre-tokenization; any other
                            // occurrence is a real word boundary.
                            if i == 0 && j == 0 && self.prepend_scheme != PrependScheme::Never {
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

    #[test]
    fn decode_first_token_with_internal_replacement() {
        // Only the single prepended replacement char should be stripped from
        // the first token; internal ones are real word boundaries.
        let decoder = MetaspaceDecoder::new('▁', PrependScheme::Always);
        let res = decoder
            .decode_chain(vec!["▁in▁the".into(), "▁house".into()])
            .unwrap();
        assert_eq!(res, vec!["in the", " house"]);

        // Same thing for a single multi-word token.
        let res = decoder.decode_chain(vec!["▁Hey▁friend".into()]).unwrap();
        assert_eq!(res, vec!["Hey friend"]);

        // With `Never`, nothing was prepended so nothing should be stripped.
        let decoder = MetaspaceDecoder::new('▁', PrependScheme::Never);
        let res = decoder
            .decode_chain(vec!["▁in▁the".into(), "▁house".into()])
            .unwrap();
        assert_eq!(res, vec![" in the", " house"]);
    }
}
