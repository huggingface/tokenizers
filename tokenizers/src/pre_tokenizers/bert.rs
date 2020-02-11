use crate::tokenizer::{Offsets, PreTokenizer, Result};
use unicode_categories::UnicodeCategories;

fn is_bert_punc(x: char) -> bool {
    char::is_ascii_punctuation(&x) || x.is_punctuation()
}

/// Split the given string as the `should_split` predicate dictates. Keep track of the offsets
fn split_on<F: Fn(&char) -> bool>(
    s: &str,
    should_split: F,
    include_split_token: bool,
) -> Vec<(String, Offsets)> {
    let mut words: Vec<(String, Offsets)> = vec![];
    let mut offset = 0;
    let mut word = Vec::with_capacity(50);
    s.chars().for_each(|c| {
        if should_split(&c) {
            if !word.is_empty() {
                let offsets = (offset - word.len(), offset);
                words.push((word.drain(0..).collect::<String>(), offsets));
            }
            if include_split_token {
                words.push((c.to_string(), (offset, offset + 1)));
            }
        } else if !should_split(&c) {
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

pub struct BertPreTokenizer;

impl PreTokenizer for BertPreTokenizer {
    fn pre_tokenize(&self, s: &str) -> Result<Vec<(String, Offsets)>> {
        let mut split_tokens = vec![];
        for (token, offsets) in split_on(&s, |c| char::is_whitespace(*c), false) {
            split_tokens.extend(
                split_on(&token, |c| is_bert_punc(*c), true)
                    .into_iter()
                    .map(|(tok, off)| (tok, (off.0 + offsets.0, off.1 + offsets.0))),
            );
        }
        Ok(split_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let pretok = BertPreTokenizer;
        let res = pretok
            .pre_tokenize("Hey friend!     How are you?!?")
            .unwrap();
        assert_eq!(
            &res,
            &[
                ("Hey".into(), (0, 3)),
                ("friend".into(), (4, 10)),
                ("!".into(), (10, 11)),
                ("How".into(), (16, 19)),
                ("are".into(), (20, 23)),
                ("you".into(), (24, 27)),
                ("?".into(), (27, 28)),
                ("!".into(), (28, 29)),
                ("?".into(), (29, 30)),
            ]
        );
    }
}
