use std::fmt;

use regex::Regex;
use serde::{Deserialize, Deserializer, Serialize};

use crate::tokenizer::{
    pattern::Invert, PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior,
};
use serde::de::{Error, Visitor};

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type")]
pub struct Whitespace {
    #[serde(default = "default_regex", skip)]
    re: Regex,
}

fn default_regex() -> Regex {
    Regex::new(r"\w+|[^\w\s]+").unwrap()
}

impl Default for Whitespace {
    fn default() -> Self {
        Self {
            re: default_regex(),
        }
    }
}

impl PreTokenizer for Whitespace {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, normalized| {
            normalized.split(Invert(&self.re), SplitDelimiterBehavior::Removed)
        })
    }
}

// manually implement deserialize because Whitespace is not a unit-struct but is
// serialized like one.
impl<'de> Deserialize<'de> for Whitespace {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(WhitespaceVisitor)
    }
}
struct WhitespaceVisitor;
impl<'de> Visitor<'de> for WhitespaceVisitor {
    type Value = Whitespace;
    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Whitespace")
    }

    fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        let maybe_type = map.next_entry::<String, String>()?;
        let maybe_type_str = maybe_type.as_ref().map(|(k, v)| (k.as_str(), v.as_str()));
        match maybe_type_str {
            Some(("type", "Whitespace")) => Ok(Whitespace::default()),
            Some((_, ty)) => Err(Error::custom(&format!("Expected Whitespace, got {}", ty))),
            None => Err(Error::custom("Expected type: Whitespace")),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct WhitespaceSplit;
impl_serde_unit_struct!(WhitespaceSplitVisitor, WhitespaceSplit);

impl PreTokenizer for WhitespaceSplit {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, normalized| {
            normalized.split(char::is_whitespace, SplitDelimiterBehavior::Removed)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OffsetReferential, OffsetType, PreTokenizer};

    #[test]
    fn basic() {
        let tests = vec![
            (
                "Hey man!",
                vec![("Hey", (0, 3)), ("man", (4, 7)), ("!", (7, 8))],
            ),
            (
                "How are you doing?",
                vec![
                    ("How", (0, 3)),
                    ("are", (4, 7)),
                    ("you", (8, 11)),
                    ("doing", (12, 17)),
                    ("?", (17, 18)),
                ],
            ),
            ("\n", vec![]),
        ];
        let pretok = Whitespace::default();
        for (s, res) in tests {
            let mut pretokenized = PreTokenizedString::from(s);
            pretok.pre_tokenize(&mut pretokenized).unwrap();
            assert_eq!(
                pretokenized
                    .get_splits(OffsetReferential::Original, OffsetType::Byte)
                    .into_iter()
                    .map(|(s, o, _)| (s, o))
                    .collect::<Vec<_>>(),
                res
            );
        }
    }

    #[test]
    fn whitespace_split() {
        let tests = vec![
            ("Hey man!", vec![("Hey", (0, 3)), ("man!", (4, 8))]),
            (
                "Hey, man, Good?",
                vec![("Hey,", (0, 4)), ("man,", (5, 9)), ("Good?", (10, 15))],
            ),
        ];
        let pretok = WhitespaceSplit;
        for (s, res) in tests {
            let mut pretokenized = PreTokenizedString::from(s);
            pretok.pre_tokenize(&mut pretokenized).unwrap();
            assert_eq!(
                pretokenized
                    .get_splits(OffsetReferential::Original, OffsetType::Byte)
                    .into_iter()
                    .map(|(s, o, _)| (s, o))
                    .collect::<Vec<_>>(),
                res
            );
        }
    }
}
