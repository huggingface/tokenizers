use crate::normalizer::Range;
use crate::tokenizer::{PreTokenizedString, PreTokenizer, Result};
use serde::{Deserialize, Serialize};

use crate::utils::macro_rules_attribute;

#[derive(Clone, Debug, PartialEq, Eq)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct FixedLength {
    #[serde(default = "default_length")]
    pub length: usize,
}

impl FixedLength {
    pub fn new(length: usize) -> Self {
        Self { length }
    }
}

fn default_length() -> usize {
    5
}

impl PreTokenizer for FixedLength {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, normalized| {
            let text = normalized.get();
            if text.is_empty() {
                return Ok(vec![]);
            }

            let mut splits = Vec::new();
            let char_positions: Vec<_> = text.char_indices().collect();
            for chunk in char_positions.chunks(self.length) {
                let start = chunk.first().map(|(i, _)| *i).unwrap_or(0);
                let end = chunk
                    .last()
                    .map(|(i, c)| i + c.len_utf8())
                    .unwrap_or(text.len());
                splits.push(
                    normalized
                        .slice(Range::Normalized(start..end))
                        .ok_or("Failed to slice normalized text")?,
                );
            }

            Ok(splits)
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
                "Hello world",
                vec![("Hello", (0, 5)), (" worl", (5, 10)), ("d", (10, 11))],
            ),
            ("Short", vec![("Short", (0, 5))]),
            ("", vec![]),
        ];
        let pretok = FixedLength { length: 5 };
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
    fn custom_length() {
        let pretok = FixedLength { length: 3 };
        let mut pretokenized = PreTokenizedString::from("Hello world");
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![
                ("Hel", (0, 3)),
                ("lo ", (3, 6)),
                ("wor", (6, 9)),
                ("ld", (9, 11)),
            ]
        );
    }

    #[test]
    fn utf8_characters() {
        let pretok = FixedLength { length: 3 };
        let mut pretokenized = PreTokenizedString::from("Hello 👋 world");
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![
                ("Hel", (0, 3)),
                ("lo ", (3, 6)),
                ("👋 w", (6, 12)),
                ("orl", (12, 15)),
                ("d", (15, 16)),
            ]
        );
    }
}
