use crate::normalizer::Range;
use crate::pipeline;
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

impl pipeline::PreTokenizer for FixedLength {
    fn pre_tokenize(&self, text: &str, out: &mut Vec<pipeline::Split>) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }

        if self.length == 0 {
            out.push(pipeline::Split {
                start: 0,
                end: text.len() as u32,
            });
            return Ok(());
        }

        // `step_by` yields the byte offset of every `length`-th char — i.e. the
        // start of each chunk. `skip(1)` turns those into the *end* of the
        // preceding chunk; the final chunk runs to `text.len()`.
        let mut start: u32 = 0;
        for (end, _) in text.char_indices().step_by(self.length).skip(1) {
            out.push(pipeline::Split {
                start,
                end: end as u32,
            });
            start = end as u32;
        }
        out.push(pipeline::Split {
            start,
            end: text.len() as u32,
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OffsetReferential, OffsetType, PreTokenizer};

    fn pretokenize(length: usize, text: &str) -> Vec<(&str, (u32, u32))> {
        let pretok = FixedLength { length };
        let mut splits = Vec::new();
        crate::pipeline::PreTokenizer::pre_tokenize(&pretok, text, &mut splits).unwrap();
        splits
            .iter()
            .map(|s| (&text[s.range()], (s.start, s.end)))
            .collect()
    }

    #[test]
    fn pipeline_basic() {
        // same expectations as the legacy `basic`/`custom_length` tests
        assert_eq!(
            pretokenize(5, "Hello world"),
            vec![("Hello", (0, 5)), (" worl", (5, 10)), ("d", (10, 11))],
        );
        assert_eq!(
            pretokenize(3, "Hello world"),
            vec![
                ("Hel", (0, 3)),
                ("lo ", (3, 6)),
                ("wor", (6, 9)),
                ("ld", (9, 11)),
            ],
        );
    }

    #[test]
    fn pipeline_utf8() {
        // chunks are counted in chars; offsets are bytes (👋 is 4 bytes)
        assert_eq!(
            pretokenize(3, "Hello 👋 world"),
            vec![
                ("Hel", (0, 3)),
                ("lo ", (3, 6)),
                ("👋 w", (6, 12)),
                ("orl", (12, 15)),
                ("d", (15, 16)),
            ],
        );
    }

    #[test]
    fn pipeline_edge_cases() {
        let empty = Vec::<(&str, (u32, u32))>::new();
        assert_eq!(pretokenize(5, ""), empty);
        // length >= char count -> one chunk
        assert_eq!(pretokenize(5, "Short"), vec![("Short", (0, 5))]);
        assert_eq!(pretokenize(10, "abc"), vec![("abc", (0, 3))]);
        // length == 0 -> whole text as a single split (no panic)
        assert_eq!(pretokenize(0, "abc"), vec![("abc", (0, 3))]);
    }

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
