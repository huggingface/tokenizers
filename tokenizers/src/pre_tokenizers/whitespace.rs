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
            .map(|captures| {
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
            .flatten()
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::Whitespace;
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
}
