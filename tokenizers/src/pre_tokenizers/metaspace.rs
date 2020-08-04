use serde::{Deserialize, Serialize};

use crate::tokenizer::{Decoder, PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior};

#[derive(Serialize, Deserialize, Clone, Debug)]
/// Replaces all the whitespaces by the provided meta character and then
/// splits on this character
#[serde(tag = "type")]
pub struct Metaspace {
    replacement: char,
    str_rep: String,
    add_prefix_space: bool,
}

impl Metaspace {
    pub fn new(replacement: char, add_prefix_space: bool) -> Self {
        Self {
            replacement,
            str_rep: replacement.to_string(),
            add_prefix_space,
        }
    }
}

impl Default for Metaspace {
    fn default() -> Self {
        Self::new('▁', true)
    }
}

impl PreTokenizer for Metaspace {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, mut normalized| {
            if self.add_prefix_space {
                normalized.prepend(&self.str_rep);
            }

            Ok(normalized
                .split(' ', SplitDelimiterBehavior::MergedWithNext)?
                .into_iter()
                .map(|mut normalized| {
                    normalized.replace(' ', &self.str_rep)?;
                    Ok(normalized)
                })
                .collect::<Result<Vec<_>>>()?)
        })
    }
}

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
    use crate::OffsetReferential;

    #[test]
    fn basic() {
        let pretok = Metaspace::new('▁', true);
        let mut pretokenized = PreTokenizedString::from("Hey friend!");
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized.get_normalized(OffsetReferential::Normalized),
            vec![("▁Hey", (0, 4)), ("▁friend!", (4, 12))]
        );
        assert_eq!(
            pretokenized.get_normalized(OffsetReferential::Original),
            vec![("▁Hey", (0, 3)), ("▁friend!", (3, 11))]
        );
    }

    #[test]
    fn multiple_spaces() {
        let pretok = Metaspace::new('▁', true);
        let mut pretokenized = PreTokenizedString::from("Hey   friend!");
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized.get_normalized(OffsetReferential::Normalized),
            vec![
                ("▁Hey", (0, 4)),
                ("▁", (4, 5)),
                ("▁", (5, 6)),
                ("▁friend!", (6, 14)),
            ]
        );
        assert_eq!(
            pretokenized.get_normalized(OffsetReferential::Original),
            vec![
                ("▁Hey", (0, 3)),
                ("▁", (3, 4)),
                ("▁", (4, 5)),
                ("▁friend!", (5, 13)),
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
