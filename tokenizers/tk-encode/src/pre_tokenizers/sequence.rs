use crate::pipeline;
use crate::pre_tokenizers::PreTokenizerWrapper;
use crate::tokenizer::{PreTokenizedString, PreTokenizer, Result};
use crate::utils::macro_rules_attribute;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct Sequence {
    pretokenizers: Vec<PreTokenizerWrapper>,
}

impl Sequence {
    pub fn new(pretokenizers: Vec<PreTokenizerWrapper>) -> Self {
        Self { pretokenizers }
    }
}

impl AsRef<[PreTokenizerWrapper]> for Sequence {
    fn as_ref(&self) -> &[PreTokenizerWrapper] {
        &self.pretokenizers
    }
}

impl AsMut<[PreTokenizerWrapper]> for Sequence {
    fn as_mut(&mut self) -> &mut [PreTokenizerWrapper] {
        &mut self.pretokenizers
    }
}

impl IntoIterator for Sequence {
    type Item = PreTokenizerWrapper;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.pretokenizers.into_iter()
    }
}

impl PreTokenizer for Sequence {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        for pretokenizer in &self.pretokenizers {
            pretokenizer.pre_tokenize(pretokenized)?;
        }
        Ok(())
    }
}

impl pipeline::PreTokenizer for Sequence {
    /// Runs each child in turn, where every child subdivides the spans produced
    /// so far. A child sees only the text of a span (`&text[span]`) and returns
    /// offsets relative to it, which we rebase to absolute mirroring how the
    /// legacy path worked.
    fn pre_tokenize(&self, text: &str, out: &mut Vec<pipeline::Split>) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }

        let cap = text.len() / 5;

        let mut current: Vec<pipeline::Split> = Vec::with_capacity(cap);
        current.push(pipeline::Split {
            start: 0,
            end: text.len() as u32,
        });
        let mut next: Vec<pipeline::Split> = Vec::with_capacity(cap);

        for child in &self.pretokenizers {
            next.clear();
            for span in &current {
                let base = span.start;
                // The child appends span-relative spans straight into `next`;
                // rebase just those to absolute in place — no scratch buffer.
                let from = next.len();
                pipeline::PreTokenizer::pre_tokenize(child, &text[span.range()], &mut next)?;
                // FIXME: do we want to add an `offset` param to `pre_tokenize` so we don't have to
                // rebase?
                for s in &mut next[from..] {
                    s.start += base;
                    s.end += base;
                }
            }
            std::mem::swap(&mut current, &mut next);
        }

        out.extend_from_slice(&current);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pre_tokenizers::byte_level::ByteLevel;
    use crate::pre_tokenizers::digits::Digits;
    use crate::pre_tokenizers::whitespace::Whitespace;
    use crate::pre_tokenizers::{punctuation::Punctuation, whitespace::WhitespaceSplit};
    use crate::{OffsetReferential, OffsetType};

    /// Run the pipeline path and return `(piece, (start, end))` for each split.
    fn pipeline_pretokenize(seq: &Sequence, text: &str) -> Vec<(String, (usize, usize))> {
        let mut out = Vec::new();
        crate::pipeline::PreTokenizer::pre_tokenize(seq, text, &mut out).unwrap();
        out.iter()
            .map(|s| {
                (
                    text[s.range()].to_string(),
                    (s.start as usize, s.end as usize),
                )
            })
            .collect()
    }

    /// The legacy path's `(piece, offsets)` — the oracle.
    fn legacy_pretokenize(seq: &Sequence, text: &str) -> Vec<(String, (usize, usize))> {
        let mut pre = PreTokenizedString::from(text);
        PreTokenizer::pre_tokenize(seq, &mut pre).unwrap();
        pre.get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(s, o, _)| (s.to_string(), o))
            .collect()
    }

    #[test]
    fn pipeline_sequence_basic() {
        // Same config + expectation as `sequence_basic`, via the range API.
        let seq = Sequence::new(vec![
            PreTokenizerWrapper::WhitespaceSplit(WhitespaceSplit),
            PreTokenizerWrapper::Punctuation(Punctuation::default()),
        ]);
        assert_eq!(
            pipeline_pretokenize(&seq, "Hey friend!     How are you?!?"),
            [
                ("Hey", (0, 3)),
                ("friend", (4, 10)),
                ("!", (10, 11)),
                ("How", (16, 19)),
                ("are", (20, 23)),
                ("you", (24, 27)),
                ("?", (27, 28)),
                ("!", (28, 29)),
                ("?", (29, 30)),
            ]
            .map(|(s, o)| (s.to_string(), o)),
        );
    }

    #[test]
    fn pipeline_matches_legacy_oracle() {
        // Differential: the pipeline path must equal the legacy path across
        // varied configs (incl. a nested Sequence) and multi-script texts.
        let configs: Vec<Vec<PreTokenizerWrapper>> = vec![
            vec![
                PreTokenizerWrapper::WhitespaceSplit(WhitespaceSplit),
                PreTokenizerWrapper::Punctuation(Punctuation::default()),
            ],
            vec![PreTokenizerWrapper::Whitespace(Whitespace)],
            vec![
                PreTokenizerWrapper::WhitespaceSplit(WhitespaceSplit),
                PreTokenizerWrapper::Digits(Digits::new(true)),
                PreTokenizerWrapper::Punctuation(Punctuation::default()),
            ],
            // nested Sequence as a child
            vec![
                PreTokenizerWrapper::Sequence(Sequence::new(vec![
                    PreTokenizerWrapper::WhitespaceSplit(WhitespaceSplit),
                ])),
                PreTokenizerWrapper::Punctuation(Punctuation::default()),
            ],
        ];
        let texts = [
            "Hey friend!     How are you?!?",
            "abc 123 def!!ghi 42",
            "  leading  and   trailing spaces  ",
            "café? no—maybe 3.14 ok",
            "中文 text 123, mixed!",
            "single",
            "!!!",
        ];
        for (ci, cfg) in configs.into_iter().enumerate() {
            let seq = Sequence::new(cfg);
            for text in texts {
                assert_eq!(
                    pipeline_pretokenize(&seq, text),
                    legacy_pretokenize(&seq, text),
                    "config #{ci} diverged on {text:?}",
                );
            }
        }
    }

    #[test]
    fn pipeline_empty_input() {
        let seq = Sequence::new(vec![
            PreTokenizerWrapper::WhitespaceSplit(WhitespaceSplit),
            PreTokenizerWrapper::Punctuation(Punctuation::default()),
        ]);
        assert!(pipeline_pretokenize(&seq, "").is_empty());
    }

    #[test]
    fn pipeline_unsupported_child_errors() {
        // A byte-rewriting child has no range-based form → the whole Sequence errors.
        let seq = Sequence::new(vec![
            PreTokenizerWrapper::WhitespaceSplit(WhitespaceSplit),
            PreTokenizerWrapper::ByteLevel(ByteLevel::default()),
        ]);
        let mut out = Vec::new();
        assert!(crate::pipeline::PreTokenizer::pre_tokenize(&seq, "hi there", &mut out).is_err());
    }

    #[test]
    fn sequence_basic() {
        let pretokenizers = vec![
            PreTokenizerWrapper::WhitespaceSplit(WhitespaceSplit),
            PreTokenizerWrapper::Punctuation(Punctuation::default()),
        ];
        let pretok = Sequence::new(pretokenizers);
        let mut pretokenized: PreTokenizedString = "Hey friend!     How are you?!?".into();
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![
                ("Hey", (0, 3)),
                ("friend", (4, 10)),
                ("!", (10, 11)),
                ("How", (16, 19)),
                ("are", (20, 23)),
                ("you", (24, 27)),
                ("?", (27, 28)),
                ("!", (28, 29)),
                ("?", (29, 30)),
            ]
        );
    }
}
