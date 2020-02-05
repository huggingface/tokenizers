use crate::tokenizer::{Offsets, PreTokenizer, Result};
use regex::Regex;

pub struct Whitespace;
impl PreTokenizer for Whitespace {
    fn pre_tokenize(&self, s: &str) -> Result<Vec<(String, Offsets)>> {
        lazy_static! {
            static ref RE: Regex = Regex::new(r"\w+|[^\w\s]+").unwrap();
        }
        Ok(RE
            .captures_iter(s)
            .flat_map(|captures| {
                captures
                    .iter()
                    .map(|m| {
                        m.map(|capture| {
                            let (start, end) = (capture.start(), capture.end());
                            (s[start..end].to_owned(), (start, end))
                        })
                        .unwrap_or_else(|| (String::from(""), (0, 0)))
                    })
                    .collect::<Vec<(String, Offsets)>>()
            })
            .collect())
    }
}

pub struct WhitespaceSplit;
impl PreTokenizer for WhitespaceSplit {
    fn pre_tokenize(&self, s: &str) -> Result<Vec<(String, Offsets)>> {
        let mut words = vec![];
        let mut word = Vec::with_capacity(1000);
        let mut offset = 0;

        s.chars().for_each(|c| {
            if c.is_whitespace() {
                if !word.is_empty() {
                    let offsets = (offset - word.len(), offset);
                    words.push((word.drain(0..).collect::<String>(), offsets));
                }
            } else {
                word.push(c);
            }
            offset += 1;
        });
        if !word.is_empty() {
            let offsets = (offset - word.len(), offset);
            words.push((word.drain(0..).collect::<String>(), offsets));
        }

        Ok(words)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::PreTokenizer;

    #[test]
    fn basic() {
        let tests = vec![
            (
                "Hey man!",
                vec![
                    ("Hey".into(), (0, 3)),
                    ("man".into(), (4, 7)),
                    ("!".into(), (7, 8)),
                ],
            ),
            (
                "How are you doing?",
                vec![
                    ("How".into(), (0, 3)),
                    ("are".into(), (4, 7)),
                    ("you".into(), (8, 11)),
                    ("doing".into(), (12, 17)),
                    ("?".into(), (17, 18)),
                ],
            ),
        ];
        let pretok = Whitespace;
        for (s, res) in tests {
            assert_eq!(pretok.pre_tokenize(s).unwrap(), res);
        }
    }

    #[test]
    fn whitespace_split() {
        let tests = vec![
            (
                "Hey man!",
                vec![("Hey".into(), (0, 3)), ("man!".into(), (4, 8))],
            ),
            (
                "Hey, man, Good?",
                vec![
                    ("Hey,".into(), (0, 4)),
                    ("man,".into(), (5, 9)),
                    ("Good?".into(), (10, 15)),
                ],
            ),
        ];
        let pretok = WhitespaceSplit;
        for (s, res) in tests {
            assert_eq!(pretok.pre_tokenize(s).unwrap(), res);
        }
    }
}
