//! The `Metaspace` **decoder**.
//!
//! There is no `Metaspace` *pre-tokenizer* to be a different type from any more: the encode half is
//! a [`MetaspaceNormalizer`] plus a `Split`, which is what reading one builds. Decoding stays a
//! component of its own — see [`crate::decoders::replace`] for why.
//!
//! `PrependScheme` lives here because this is the only component left that holds one; the JSON layer
//! reads it for both halves. It is plain data describing a setting, not a component playing a role.
//!
//! [`MetaspaceNormalizer`]: crate::normalizers::metaspace::MetaspaceNormalizer

use crate::tokenizer::{Decoder, Result};

/// Enum representing options for the metaspace prepending scheme.
///
/// The JSON spelling is `snake_case`: `"first"` / `"never"` / `"always"`.
#[derive(Debug, Clone, PartialEq, Eq, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum PrependScheme {
    /// Specifies that the scheme should be prepended only once, on the first split.
    First,
    /// Specifies that the space should not be prepended.
    Never,
    /// Specifies that the scheme should always be prepended.
    Always,
}

impl std::fmt::Display for PrependScheme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Spelled out rather than handed to the serializer, so the name survives a build with no
        // serde in it. These must stay identical to the `rename_all = "snake_case"` spelling above;
        // `display_matches_serde` pins that.
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

    /// `PrependScheme`'s `Display` is spelled out by hand so the name survives a build with no serde
    /// in it. This is what stops it drifting from the `rename_all = "snake_case"` derive.
    #[test]
    #[cfg(feature = "serde")]
    fn display_matches_serde() {
        for scheme in [
            PrependScheme::First,
            PrependScheme::Never,
            PrependScheme::Always,
        ] {
            let via_serde = serde_json::to_string(&scheme).unwrap();
            assert_eq!(format!("\"{scheme}\""), via_serde);
        }
    }

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
