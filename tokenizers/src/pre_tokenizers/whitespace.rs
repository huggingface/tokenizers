use onig::Regex;

pub struct WhitespaceTokenizer();

impl WhitespaceTokenizer {
    pub fn tokenize(s: &str) -> Vec<String> {
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
    use super::WhitespaceTokenizer;

    #[test]
    fn separate_words() {
        let tests = vec![
            ("Hey man!", vec!["Hey", "man", "!"]),
            (
                "How are you doing?",
                vec!["How", "are", "you", "doing", "?"],
            ),
        ];
        for (s, res) in tests {
            assert_eq!(WhitespaceTokenizer::tokenize(s), res);
        }
    }
}
