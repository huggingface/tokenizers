use crate::tokenizer::{Decoder, PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
/// Replaces all the whitespaces by the provided meta character and then
/// splits on this character
pub struct Metaspace {
    replacement: char,
    str_bytes: [u8; 4],
    add_prefix_space: bool,
}

impl Metaspace {
    pub fn new(replacement: char, add_prefix_space: bool) -> Self {
        let mut str_bytes = [0; 4];
        replacement.encode_utf8(&mut str_bytes);
        Self {
            replacement,
            str_bytes,
            add_prefix_space,
        }
    }

    #[inline]
    fn replacement(&self) -> &str {
        unsafe { std::str::from_utf8_unchecked(&self.str_bytes[..self.replacement.len_utf8()]) }
    }
}

impl Default for Metaspace {
    fn default() -> Self {
        Self::new('▁', true)
    }
}

#[typetag::serde]
impl PreTokenizer for Metaspace {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        let replacement = self.replacement.to_string();
        pretokenized.split(|_, normalized| {
            normalized
                .split(' ', SplitDelimiterBehavior::MergedWithNext)
                .into_iter()
                .enumerate()
                .map(|(i, mut normalized)| {
                    normalized.replace(' ', self.replacement());
                    if i > 0 || (i == 0 && self.add_prefix_space) {
                        normalized.prepend(&replacement);
                    }
                    normalized
                })
        })
    }
}

#[typetag::serde]
impl Decoder for Metaspace {
    fn decode(&self, tokens: Vec<String>) -> Result<String> {
        Ok(tokens
            .iter()
            .flat_map(|t| t.chars())
            .enumerate()
            .filter_map(|(i, c)| {
                if c == self.replacement {
                    if i == 0 && self.add_prefix_space {
                        None
                    } else {
                        Some(' ')
                    }
                } else {
                    Some(c)
                }
            })
            .collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let pretok = Metaspace::new('▁', true);
        let mut pretokenized = PreTokenizedString::from("Hey friend!");
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized.get_normalized(),
            vec![("▁Hey", (0, 4)), ("▁friend!", (4, 12))]
        );
    }

    #[test]
    fn multiple_spaces() {
        let pretok = Metaspace::new('▁', true);
        let mut pretokenized = PreTokenizedString::from("Hey   friend!");
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized.get_normalized(),
            vec![
                ("▁Hey", (0, 4)),
                ("▁", (4, 5)),
                ("▁", (5, 6)),
                ("▁friend!", (6, 14)),
            ]
        );
    }

    #[test]
    fn decode() {
        let decoder = Metaspace::new('▁', true);
        let res = decoder
            .decode(vec!["▁Hey".into(), "▁friend!".into()])
            .unwrap();
        assert_eq!(&res, "Hey friend!")
    }
}
