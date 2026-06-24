use crate::pipeline;
use crate::tokenizer::{PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior};
use crate::utils::macro_rules_attribute;
use unicode_categories::UnicodeCategories;

fn is_bert_punc(x: char) -> bool {
    char::is_ascii_punctuation(&x) || x.is_punctuation()
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct BertPreTokenizer;

impl PreTokenizer for BertPreTokenizer {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, s| s.split(char::is_whitespace, SplitDelimiterBehavior::Removed))?;
        pretokenized.split(|_, s| s.split(is_bert_punc, SplitDelimiterBehavior::Isolated))
    }
}

#[derive(PartialEq, Eq)]
enum CharType {
    Whitespace,
    Punctuation,
    Other,
}

impl pipeline::PreTokenizer for BertPreTokenizer {
    fn pre_tokenize(&self, text: &str, out: &mut Vec<pipeline::Split>) -> Result<()> {
        let mut start = 0;
        let mut prev_type = None;
        for (i, c) in text.char_indices() {
            let ct = if c.is_whitespace() {
                CharType::Whitespace
            } else if is_bert_punc(c) {
                CharType::Punctuation
            } else {
                CharType::Other
            };

            if let Some(pt) = prev_type {
                if pt != ct || ct == CharType::Punctuation {
                    if pt != CharType::Whitespace {
                        out.push(pipeline::Split { range: start..i });
                    }
                    start = i;
                }
            }
            prev_type = Some(ct);
        }
        if let Some(pt) = prev_type {
            if pt != CharType::Whitespace {
                out.push(pipeline::Split {
                    range: start..text.len(),
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::BertPreTokenizer;
    use crate::{NormalizedString, OffsetReferential, OffsetType, PreTokenizedString};

    #[test]
    fn basic_new() {
        use crate::pipeline::PreTokenizer;

        let pretok = BertPreTokenizer;
        let pretokenized: &str = "   Hey friend!     How are you?!?  ";
        let mut ranges = Vec::new();
        pretok.pre_tokenize(pretokenized, &mut ranges).unwrap();
        let got = ranges.into_iter().map(|s| s.range).collect::<Vec<_>>();
        assert_eq!(
            got,
            vec![
                3..6,
                7..13,
                13..14,
                19..22,
                23..26,
                27..30,
                30..31,
                31..32,
                32..33,
            ]
        );
    }

    #[test]
    fn chinese_chars_new() {
        use crate::pipeline::PreTokenizer;

        let mut n = NormalizedString::from("野口里佳 Noguchi Rika");
        n.transform(
            n.get().to_owned().chars().flat_map(|c| {
                if (c as usize) > 0x4E00 {
                    vec![(' ', 0), (c, 1), (' ', 1)]
                } else {
                    vec![(c, 0)]
                }
            }),
            0,
        );
        let pretokenized: &str = n.get();
        let pretok = BertPreTokenizer;
        let mut ranges = Vec::new();
        pretok.pre_tokenize(pretokenized, &mut ranges).unwrap();
        let got = ranges.into_iter().map(|s| s.range).collect::<Vec<_>>();
        assert_eq!(got, vec![1..4, 6..9, 11..14, 16..19, 21..28, 29..33]);
    }

    #[test]
    fn basic() {
        use crate::PreTokenizer;

        let pretok = BertPreTokenizer;
        let mut pretokenized: PreTokenizedString = "Hey friend!     How are you?!?".into();
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![
                ("Hey", (0, 3)),
                ("friend", (4, 10)),
                ("!", (10, 11)),
                ("How", (16, 19)),
                ("are", (20, 23)),
                ("you", (24, 27)),
                ("?", (27, 28)),
                ("!", (28, 29)),
                ("?", (29, 30)),
            ]
        );
    }

    #[test]
    fn chinese_chars() {
        use crate::PreTokenizer;

        let mut n = NormalizedString::from("野口里佳 Noguchi Rika");
        n.transform(
            n.get().to_owned().chars().flat_map(|c| {
                if (c as usize) > 0x4E00 {
                    vec![(' ', 0), (c, 1), (' ', 1)]
                } else {
                    vec![(c, 0)]
                }
            }),
            0,
        );
        let mut pretokenized = n.into();
        let pretok = BertPreTokenizer;
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![
                ("野", (0, 3)),
                ("口", (3, 6)),
                ("里", (6, 9)),
                ("佳", (9, 12)),
                ("Noguchi", (13, 20)),
                ("Rika", (21, 25))
            ]
        );
    }
}
