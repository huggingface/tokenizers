use crate::tokenizer::{PreTokenizer, Result};
use regex::Regex;

pub struct Whitespace;
impl PreTokenizer for Whitespace {
    fn pre_tokenize(&self, s: &str) -> Result<Vec<String>> {
        lazy_static! {
            static ref RE: Regex = Regex::new(r"\w+|[^\w\s]+").unwrap();
        }
        Ok(RE
            .captures_iter(s)
            .map(|captures| {
                captures
                    .iter()
                    .map(|m| {
                        m.map(|capture| s[capture.start()..capture.end()].to_owned())
                            .unwrap_or_else(|| String::from(""))
                    })
                    .collect()
            })
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
            ("Hey man!", vec!["Hey", "man", "!"]),
            (
                "How are you doing?",
                vec!["How", "are", "you", "doing", "?"],
            ),
        ];
        let pretok = Whitespace;
        for (s, res) in tests {
            assert_eq!(pretok.pre_tokenize(s).unwrap(), res);
        }
    }
}
