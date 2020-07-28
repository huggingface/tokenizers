use crate::tokenizer::{Offsets, PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior};
use serde::{Deserialize, Serialize};
use unicode_categories::UnicodeCategories;

fn is_bert_punc(x: char) -> bool {
    char::is_ascii_punctuation(&x) || x.is_punctuation()
}

/// Split the given string as the `should_split` predicate dictates. Keep track of the offsets
fn split_on<F: Fn(char) -> bool>(
    s: &str,
    should_split: F,
    include_split_token: bool,
) -> Vec<(String, Offsets)> {
    let mut words: Vec<(String, Offsets)> = vec![];
    let mut offset = 0;
    let mut word = Vec::with_capacity(50);
    s.chars().for_each(|c| {
        if should_split(c) {
            if !word.is_empty() {
                let offsets = (offset - word.len(), offset);
                words.push((word.drain(0..).collect::<String>(), offsets));
            }
            if include_split_token {
                words.push((c.to_string(), (offset, offset + 1)));
            }
        } else if !should_split(c) {
            word.push(c);
        }
        offset += 1;
    });
    // Don't forget the potential last word
    if !word.is_empty() {
        let offsets = (offset - word.len(), offset);
        words.push((word.drain(0..).collect::<String>(), offsets));
    }

    words
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

        // let mut split_tokens = vec![];
        // for (token, offsets) in split_on(normalized.get(), char::is_whitespace, false) {
        //     split_tokens.extend(
        //         split_on(&token, is_bert_punc, true)
        //             .into_iter()
        //             .map(|(tok, off)| (tok, (off.0 + offsets.0, off.1 + offsets.0))),
        //     );
        // }
        // Ok(split_tokens)
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
}
