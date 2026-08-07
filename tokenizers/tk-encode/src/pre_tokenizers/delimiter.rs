use serde::{Deserialize, Serialize};

use crate::pipeline;
use crate::pipeline::PreTokenizerScratch;
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

// SAFETY: the spans come from `atomsplit::fsm::CharDelimiterSplit`, which splits only at character
// boundaries of `text`. It scans for the delimiter's own UTF-8 bytes and confirms the whole encoding
// before cutting.
unsafe impl pipeline::PreTokenizer for CharDelimiterSplit {
    fn pre_tokenize(
        &self,
        text: &str,
        scratch: &mut PreTokenizerScratch,
        out: &mut Vec<pipeline::Span>,
    ) -> Result<()> {
        // native atomsplit FSM (memchr-backed single-byte scan); `Removed` — drops the delimiter,
        // keeps the runs between, no empty spans. Byte-exact with the char-predicate split.
        // It keys on the delimiter's own bytes rather than on an atom class, so it needs no tags.
        scratch.split_on_bytes(
            text.as_bytes(),
            |bytes, spans| {
                atomsplit::fsm::CharDelimiterSplit(self.delimiter).pre_tokenize(
                    bytes,
                    &mut [],
                    spans,
                )
            },
            out,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pretokenize(delimiter: char, text: &str) -> Vec<(&str, (u32, u32))> {
        let pretok = CharDelimiterSplit::new(delimiter);
        let mut scratch = pipeline::PreTokenizerScratch::default();
        let mut splits = Vec::new();
        crate::pipeline::PreTokenizer::pre_tokenize(&pretok, text, &mut scratch, &mut splits)
            .unwrap();
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
