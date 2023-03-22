use serde::{Deserialize, Deserializer, Serialize};

use crate::tokenizer::{Decoder, PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior};

#[derive(Debug, Clone, PartialEq, Serialize, Eq)]
/// Replaces all the whitespaces by the provided meta character and then
/// splits on this character
#[serde(tag = "type")]
pub struct Metaspace {
    replacement: char,
    pub add_prefix_space: bool,
    pub split: bool,
    #[serde(skip)]
    str_rep: String,
}

fn default_true() -> bool {
    true
}

impl<'de> Deserialize<'de> for Metaspace {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        enum Type {
            Metaspace,
        }

        #[derive(Deserialize)]
        pub struct MetaspaceHelper {
            #[serde(rename = "type")]
            _type: Type,
            replacement: char,
            pub add_prefix_space: bool,
            #[serde(default = "default_true")]
            split: bool,
            #[serde(skip, rename = "str_rep")]
            _str_rep: String,
        }

        let helper = MetaspaceHelper::deserialize(deserializer)?;
        Ok(Self::new(
            helper.replacement,
            helper.add_prefix_space,
            helper.split,
        ))
    }
}

impl Metaspace {
    pub fn new(replacement: char, add_prefix_space: bool, split: bool) -> Self {
        Self {
            replacement,
            str_rep: replacement.to_string(),
            add_prefix_space,
            split,
        }
    }

    pub fn get_replacement(&self) -> char {
        self.replacement
    }

    pub fn set_replacement(&mut self, replacement: char) {
        self.replacement = replacement;
        self.str_rep = replacement.to_string();
    }
}

impl Default for Metaspace {
    fn default() -> Self {
        Self::new('▁', true, true)
    }
}

impl PreTokenizer for Metaspace {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, mut normalized| {
            normalized.replace(' ', &self.str_rep)?;
            if self.add_prefix_space && !normalized.get().starts_with(self.replacement) {
                normalized.prepend(&self.str_rep);
            }
            if self.split {
                normalized.split(self.replacement, SplitDelimiterBehavior::MergedWithNext)
            } else {
                normalized.split('\0', SplitDelimiterBehavior::MergedWithNext)
            }
        })
    }
}

impl Decoder for Metaspace {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        let result = Ok(tokens
            .iter()
            .enumerate()
            .map(|(i, token)| {
                token
                    .chars()
                    .enumerate()
                    .flat_map(|(j, c)| {
                        if c == self.replacement {
                            if i == 0 && j == 0 && self.add_prefix_space {
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
            .collect());
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OffsetReferential, OffsetType};

    #[test]
    fn serialization() {
        let metaspace = Metaspace::new('_', true, true);
        let metaspace_s =
            r#"{"type":"Metaspace","replacement":"_","add_prefix_space":true,"split":true}"#;
        assert_eq!(serde_json::to_string(&metaspace).unwrap(), metaspace_s);
        assert_eq!(
            serde_json::from_str::<Metaspace>(metaspace_s).unwrap(),
            metaspace
        );

        // Version without split
        let metaspace_s = r#"{"type":"Metaspace","replacement":"_","add_prefix_space":true}"#;
        let metaspace = Metaspace::new('_', true, true);
        assert_eq!(
            serde_json::from_str::<Metaspace>(metaspace_s).unwrap(),
            metaspace
        );

        // Also check it can deserialize previous versions
        let metaspace = Metaspace::new('_', true, true);
        let metaspace_s =
            r#"{"type":"Metaspace","str_rep":"_","replacement":"_","add_prefix_space":true}"#;
        assert_eq!(
            serde_json::from_str::<Metaspace>(metaspace_s).unwrap(),
            metaspace
        );

        let metaspace_parsed: Metaspace = serde_json::from_str(
            r#"{"type":"Metaspace","replacement":"_","add_prefix_space":true}"#,
        )
        .unwrap();
        assert_eq!(metaspace_parsed, metaspace);
    }

    #[test]
    fn basic() {
        let pretok = Metaspace::new('▁', true, true);
        let mut pretokenized = PreTokenizedString::from("Hey friend!");
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Normalized, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![("▁Hey", (0, 6)), ("▁friend!", (6, 16))]
        );
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![("▁Hey", (0, 3)), ("▁friend!", (3, 11))]
        );
    }

    #[test]
    fn multiple_spaces() {
        let pretok = Metaspace::new('▁', true, true);
        let mut pretokenized = PreTokenizedString::from("Hey   friend!");
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Normalized, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![
                ("▁Hey", (0, 6)),
                ("▁", (6, 9)),
                ("▁", (9, 12)),
                ("▁friend!", (12, 22)),
            ]
        );
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![
                ("▁Hey", (0, 3)),
                ("▁", (3, 4)),
                ("▁", (4, 5)),
                ("▁friend!", (5, 13)),
            ]
        );
    }

    #[test]
    fn multiple_spaces_nosplit() {
        let pretok = Metaspace::new('▁', true, false);
        let mut pretokenized = PreTokenizedString::from("Hey   friend!");
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Normalized, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![("▁Hey▁▁▁friend!", (0, 22))]
        );
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![("▁Hey▁▁▁friend!", (0, 13))]
        );
    }

    // #[test]
    // fn multiple_spaces_no_split() {
    //     let pretok = Metaspace::new('▁', true, false);
    //     let mut pretokenized = PreTokenizedString::from("Hey   friend!");
    //     pretok.pre_tokenize(&mut pretokenized).unwrap();
    //     assert_eq!(
    //         pretokenized
    //             .get_splits(OffsetReferential::Normalized, OffsetType::Byte)
    //             .into_iter()
    //             .map(|(s, o, _)| (s, o))
    //             .collect::<Vec<_>>(),
    //         vec![]
    //     );
    //     assert_eq!(
    //         pretokenized
    //             .get_splits(OffsetReferential::Original, OffsetType::Byte)
    //             .into_iter()
    //             .map(|(s, o, _)| (s, o))
    //             .collect::<Vec<_>>(),
    //         vec![
    //         ]
    //     );
    // }

    #[test]
    fn decode() {
        let decoder = Metaspace::new('▁', true, true);
        let res = decoder
            .decode_chain(vec!["▁Hey".into(), "▁friend!".into()])
            .unwrap();
        assert_eq!(res, vec!["Hey", " friend!"])
    }

    #[test]
    fn decode_two() {
        let decoder = Metaspace::new('▁', true, true);
        let res = decoder
            .decode_chain(vec!["▁▁Hey".into(), "▁friend!".into()])
            .unwrap();
        // Extra space
        assert_eq!(res, vec![" Hey", " friend!"])
    }
}
