use crate::tokenizer::{PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior};
use unicode_categories::UnicodeCategories;

fn is_bert_punc(x: char) -> bool {
    char::is_ascii_punctuation(&x) || x.is_punctuation()
}

#[derive(Copy, Clone, Debug)]
pub struct BertPreTokenizer;
impl_serde_unit_struct!(BertVisitor, BertPreTokenizer);

impl PreTokenizer for BertPreTokenizer {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, s| s.split(char::is_whitespace, SplitDelimiterBehavior::Removed))?;
        pretokenized.split(|_, s| s.split(is_bert_punc, SplitDelimiterBehavior::Isolated))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NormalizedString, OffsetReferential, OffsetType};

    #[test]
    fn basic() {
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
