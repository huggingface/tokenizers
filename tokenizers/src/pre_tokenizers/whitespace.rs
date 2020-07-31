use crate::tokenizer::{
    pattern::Invert, PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior,
};
use regex::Regex;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Whitespace {
    #[serde(default = "default_regex", skip)]
    re: Regex,
}

fn default_regex() -> Regex {
    Regex::new(r"\w+|[^\w\s]+").unwrap()
}

impl Default for Whitespace {
    fn default() -> Self {
        Self {
            re: default_regex(),
        }
    }
}

#[typetag::serde]
impl PreTokenizer for Whitespace {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, normalized| {
            normalized.split(Invert(&self.re), SplitDelimiterBehavior::Removed)
            //RE.find_iter(&normalized.get())
            //    .map(|m| {
            //        let (start, end) = (m.start(), m.end());
            //        println!("{:?}\t{:?}", start, end);
            //        normalized
            //            .slice_bytes(Range::Normalized(start..end))
            //            .expect("Whitespace cannot split according to regex")
            //    })
            //    .collect::<Vec<_>>()
        })
    }
}

#[derive(Serialize, Deserialize)]
pub struct WhitespaceSplit;
#[typetag::serde]
#[deprecated = "Prefer using DelimiterSplit, specifying the relevant delimiter"]
impl PreTokenizer for WhitespaceSplit {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, normalized| normalized.split(' ', SplitDelimiterBehavior::Removed))
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
                vec![("Hey", (0, 3)), ("man", (4, 7)), ("!", (7, 8))],
            ),
            (
                "How are you doing?",
                vec![
                    ("How", (0, 3)),
                    ("are", (4, 7)),
                    ("you", (8, 11)),
                    ("doing", (12, 17)),
                    ("?", (17, 18)),
                ],
            ),
        ];
        let pretok = Whitespace::default();
        for (s, res) in tests {
            let mut pretokenized = PreTokenizedString::from(s);
            pretok.pre_tokenize(&mut pretokenized).unwrap();
            assert_eq!(pretokenized.get_normalized(), res);
        }
    }

    #[test]
    fn whitespace_split() {
        let tests = vec![
            ("Hey man!", vec![("Hey", (0, 3)), ("man!", (4, 8))]),
            (
                "Hey, man, Good?",
                vec![("Hey,", (0, 4)), ("man,", (5, 9)), ("Good?", (10, 15))],
            ),
        ];
        let pretok = WhitespaceSplit;
        for (s, res) in tests {
            let mut pretokenized = PreTokenizedString::from(s);
            pretok.pre_tokenize(&mut pretokenized).unwrap();
            assert_eq!(pretokenized.get_normalized(), res);
        }
    }
}
