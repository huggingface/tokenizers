use crate::tokenizer::{PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior};
use serde::{Deserialize, Serialize};
use unicode_categories::UnicodeCategories;

fn is_bert_punc(x: char) -> bool {
    char::is_ascii_punctuation(&x) || x.is_punctuation()
}

#[derive(Serialize, Deserialize)]
pub struct BertPreTokenizer;

#[typetag::serde]
impl PreTokenizer for BertPreTokenizer {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, sub| {
            Ok(sub
                .split(char::is_whitespace, SplitDelimiterBehavior::Removed)?
                .into_iter()
                .flat_map(|sub| {
                    let result = sub.split(is_bert_punc, SplitDelimiterBehavior::Isolated);
                    if let Err(e) = result {
                        itertools::Either::Left(std::iter::once(Err(e)))
                    } else {
                        itertools::Either::Right(result.unwrap().into_iter().map(Ok))
                    }
                })
                .collect::<Result<Vec<_>>>()?)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let pretok = BertPreTokenizer;
        let mut pretokenized: PreTokenizedString = "Hey friend!     How are you?!?".into();
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized.get_normalized(),
            vec![
                ("Hey", (0, 3)),
                ("", (3, 4)),
                ("friend", (4, 10)),
                ("!", (10, 11)),
                ("", (11, 12)),
                ("", (12, 13)),
                ("", (13, 14)),
                ("", (14, 15)),
                ("", (15, 16)),
                ("How", (16, 19)),
                ("", (19, 20)),
                ("are", (20, 23)),
                ("", (23, 24)),
                ("you", (24, 27)),
                ("?", (27, 28)),
                ("!", (28, 29)),
                ("?", (29, 30)),
            ]
        );
    }
}
