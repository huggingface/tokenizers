use serde::{Deserialize, Serialize};

use crate::pipeline::{self, PreTokenizerScratch};
use crate::tokenizer::{PreTokenizer, Result, SplitDelimiterBehavior};
use crate::utils::macro_rules_attribute;
use SplitDelimiterBehavior::{Isolated, Removed};
use atomsplit::classify::mask;
use atomsplit::fsm::class_runs_into;
use unicode_categories::UnicodeCategories;

pub(crate) fn is_punc(x: char) -> bool {
    char::is_ascii_punctuation(&x) || x.is_punctuation()
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct Punctuation {
    #[serde(default = "default_split")]
    pub behavior: SplitDelimiterBehavior,
}

fn default_split() -> SplitDelimiterBehavior {
    SplitDelimiterBehavior::Isolated
}

impl Punctuation {
    pub fn new(behavior: SplitDelimiterBehavior) -> Self {
        Self { behavior }
    }
}

impl Default for Punctuation {
    fn default() -> Self {
        Self::new(SplitDelimiterBehavior::Isolated)
    }
}

impl PreTokenizer for Punctuation {}

// SAFETY: both routes cut only at character boundaries of `text`.
// The class-runs route is an `atomsplit` fsm.
// the merge behaviors go through `pipeline::split_delimiter`, which takes its offsets from `str::char_indices`.
unsafe impl pipeline::PreTokenizer for Punctuation {
    fn pre_tokenize(
        &self,
        text: &str,
        scratch: &mut PreTokenizerScratch,
        out: &mut Vec<pipeline::Span>,
    ) -> Result<()> {
        // atom `PUNCT` == `is_punc` (ASCII-punct ∪ \p{P}), so Isolated/Removed map to the class-runs FSM
        // byte-exactly. The merge/contiguous behaviors aren't a class-runs shape → keep the scalar split.
        if self.behavior == Removed {
            // drop punct
            scratch.split_on_tags(
                text.as_bytes(),
                class_runs_into::<{ mask::PUNCT }, 0, 0>,
                out,
            );
        } else if self.behavior == Isolated {
            // isolate each punct
            scratch.split_on_tags(
                text.as_bytes(),
                class_runs_into::<0, { mask::PUNCT }, 0>,
                out,
            );
        } else {
            pipeline::split_delimiter(text, out, is_punc, self.behavior);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pretokenize(behavior: SplitDelimiterBehavior, text: &str) -> Vec<(&str, (u32, u32))> {
        let pretok = Punctuation::new(behavior);
        let mut splits = Vec::new();
        let mut scratch = PreTokenizerScratch::default();
        crate::pipeline::PreTokenizer::pre_tokenize(&pretok, text, &mut scratch, &mut splits)
            .unwrap();
        splits
            .iter()
            .map(|s| (&text[s.range()], (s.start, s.end)))
            .collect()
    }

    #[test]
    fn pipeline_behaviors() {
        use SplitDelimiterBehavior::*;
        // '-' is ASCII punctuation; these mirror the `SplitDelimiterBehavior` docs.
        let text = "the-final--countdown";
        assert_eq!(
            pretokenize(Isolated, text),
            vec![
                ("the", (0, 3)),
                ("-", (3, 4)),
                ("final", (4, 9)),
                ("-", (9, 10)),
                ("-", (10, 11)),
                ("countdown", (11, 20)),
            ],
        );
        assert_eq!(
            pretokenize(Removed, text),
            vec![("the", (0, 3)), ("final", (4, 9)), ("countdown", (11, 20))],
        );
        assert_eq!(
            pretokenize(Contiguous, text),
            vec![
                ("the", (0, 3)),
                ("-", (3, 4)),
                ("final", (4, 9)),
                ("--", (9, 11)),
                ("countdown", (11, 20)),
            ],
        );
        assert_eq!(
            pretokenize(MergedWithPrevious, text),
            vec![
                ("the-", (0, 4)),
                ("final-", (4, 10)),
                ("-", (10, 11)),
                ("countdown", (11, 20)),
            ],
        );
        assert_eq!(
            pretokenize(MergedWithNext, text),
            vec![
                ("the", (0, 3)),
                ("-final", (3, 9)),
                ("-", (9, 10)),
                ("-countdown", (10, 20)),
            ],
        );
    }

    #[test]
    fn pipeline_matches_legacy_default() {
        // default (Isolated) must agree with the legacy `punctuation_basic` expectation
        assert_eq!(
            pretokenize(
                SplitDelimiterBehavior::Isolated,
                "Hey friend!     How are you?!?"
            ),
            vec![
                ("Hey friend", (0, 10)),
                ("!", (10, 11)),
                ("     How are you", (11, 27)),
                ("?", (27, 28)),
                ("!", (28, 29)),
                ("?", (29, 30)),
            ],
        );
    }

    #[test]
    fn pipeline_edge_cases() {
        use SplitDelimiterBehavior::*;
        let empty = Vec::<(&str, (u32, u32))>::new();
        assert_eq!(pretokenize(Isolated, ""), empty);
        assert_eq!(pretokenize(Isolated, "!"), vec![("!", (0, 1))]);
        assert_eq!(pretokenize(Removed, "!"), empty);
        // leading delimiter has no previous run to merge into
        assert_eq!(
            pretokenize(MergedWithPrevious, "-a"),
            vec![("-", (0, 1)), ("a", (1, 2))],
        );
        // trailing delimiter has no following run to merge into
        assert_eq!(
            pretokenize(MergedWithNext, "a-"),
            vec![("a", (0, 1)), ("-", (1, 2))],
        );
        // multibyte: é is 2 bytes, offsets are byte offsets
        assert_eq!(
            pretokenize(Isolated, "café!"),
            vec![("café", (0, 5)), ("!", (5, 6))],
        );
    }

    #[test]
    fn deserialization() {
        let punctuation: Punctuation = serde_json::from_str(r#"{"type": "Punctuation"}"#).unwrap();
        assert_eq!(punctuation, Punctuation::default());
        assert_eq!(
            punctuation,
            Punctuation::new(SplitDelimiterBehavior::Isolated)
        );
    }

    #[test]
    #[should_panic]
    fn deserialization_erroneous() {
        let _punctuation: Punctuation =
            serde_json::from_str(r#"{"type": "WhitespaceSplit"}"#).unwrap();
    }
}
