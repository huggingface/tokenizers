use crate::tokenizer::PreTokenizer;
use onig::Regex;

pub struct Whitespace;
impl PreTokenizer for Whitespace {
    fn pre_tokenize(&self, s: &str) -> Vec<String> {
        lazy_static! {
            static ref RE: Regex = Regex::new(r"\w+|[^\w\s]+").unwrap();
        }
        RE.find_iter(s)
            .map(|(start, end)| s[start..end].to_owned())
            .collect()
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
            assert_eq!(pretok.pre_tokenize(s), res);
        }
    }
}
