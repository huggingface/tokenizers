use serde::{Deserialize, Serialize};

use crate::pipeline;
use crate::tokenizer::{PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior};
use crate::utils::macro_rules_attribute;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
#[macro_rules_attribute(impl_serde_type!)]
pub struct CharDelimiterSplit {
    pub delimiter: char,
}

impl CharDelimiterSplit {
    pub fn new(delimiter: char) -> Self {
        Self { delimiter }
    }
}

impl PreTokenizer for CharDelimiterSplit {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        // TODO: Maybe add the option to specify the behavior
        pretokenized.split(|_, normalized| {
            normalized.split(self.delimiter, SplitDelimiterBehavior::Removed)
        })
    }
}

impl pipeline::PreTokenizer for CharDelimiterSplit {
    fn pre_tokenize(&self, text: &str, out: &mut Vec<pipeline::Split>) -> Result<()> {
        let delim = self.delimiter;
        pipeline::split(
            text,
            out,
            |c| c == delim,
            |is_delim| {
                if is_delim {
                    pipeline::SplitPolicy::Remove
                } else {
                    pipeline::SplitPolicy::Keep
                }
            },
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pretokenize(delimiter: char, text: &str) -> Vec<(&str, (u32, u32))> {
        let pretok = CharDelimiterSplit::new(delimiter);
        let mut splits = Vec::new();
        crate::pipeline::PreTokenizer::pre_tokenize(&pretok, text, &mut splits).unwrap();
        splits
            .iter()
            .map(|s| (&text[s.range()], (s.start, s.end)))
            .collect()
    }

    #[test]
    fn pipeline_basic() {
        // the delimiter is dropped; runs between delimiters are kept whole
        assert_eq!(
            pretokenize('-', "a-b-c"),
            vec![("a", (0, 1)), ("b", (2, 3)), ("c", (4, 5))],
        );
    }

    #[test]
    fn pipeline_edge_cases() {
        let empty = Vec::<(&str, (u32, u32))>::new();
        assert_eq!(pretokenize('-', ""), empty);
        // only delimiters -> nothing (no empty splits)
        assert_eq!(pretokenize('-', "--"), empty);
        // leading / trailing delimiters are dropped, no empty splits
        assert_eq!(pretokenize('-', "-a-"), vec![("a", (1, 2))]);
        // consecutive delimiters collapse (the empty span between them is dropped)
        assert_eq!(pretokenize('-', "a--b"), vec![("a", (0, 1)), ("b", (3, 4))]);
        // no delimiter -> whole string is one split
        assert_eq!(pretokenize('-', "abc"), vec![("abc", (0, 3))]);
    }

    #[test]
    fn pipeline_multibyte() {
        // offsets are byte offsets; é is 2 bytes so "café" spans 0..5
        assert_eq!(
            pretokenize(' ', "café résumé"),
            vec![("café", (0, 5)), ("résumé", (6, 14))],
        );
    }
}
