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

    /// Recognize deepseek's `[Split(\p{N}{1,3}), Split(CJK), Split(big)]` prefix (each Isolated,
    /// non-inverted) — a trailing `ByteLevel` byte-map may follow. When it matches, the whole split
    /// collapses to `atomsplit::fsm::fsm_deepseek` (byte-exact with running the three Splits in turn).
    fn is_deepseek(&self) -> bool {
        use crate::pre_tokenizers::split::SplitPattern;
        use crate::tokenizer::SplitDelimiterBehavior::Isolated;
        let regex = |i: usize| match self.pretokenizers.get(i) {
            Some(PreTokenizerWrapper::Split(s))
                if s.behavior == Isolated && !s.invert =>
            {
                match &s.pattern {
                    SplitPattern::Regex(r) => Some(r.as_str()),
                    SplitPattern::String(_) => None,
                }
            }
            _ => None,
        };
        matches!(
            (regex(0), regex(1), regex(2)),
            (Some(a), Some(b), Some(c)) if crate::utils::is_deepseek(a, b, c)
        )
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

        // deepseek's 3-Split composition → one native FSM pass (also lets the Sequence handle the
        // trailing byte-map ByteLevel, which the generic child loop can't range-split).
        if self.is_deepseek() {
            use atomsplit::classify::{classify, Atoms};
            let bytes = text.as_bytes();
            let mut tags = vec![0u8; bytes.len()];
            classify::<Atoms>(bytes, &mut tags);
            let mut spans = vec![(0u32, 0u32); bytes.len() + 1];
            let n = atomsplit::fsm::fsm_deepseek(bytes, &tags, &mut spans);
            out.extend(spans[..n].iter().map(|&(s, e)| pipeline::Split { start: s, end: e }));
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
    fn pipeline_deepseek_uses_fsm_and_matches_legacy() {
        // Load deepseek-v4's real pre_tokenizer, rebuild a Sequence of just its 3 Splits (drop the
        // trailing byte-map ByteLevel), and prove: (1) the exact fixture patterns are recognized,
        // (2) the fsm_deepseek pipeline output == the 3-regex-split legacy output, byte-for-byte.
        let path = "../data/deepseek-v4-flash-base-tokenizer.json";
        if !std::path::Path::new(path).exists() {
            return; // fixture not downloaded in this environment
        }
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let splits: Vec<PreTokenizerWrapper> = v["pre_tokenizer"]["pretokenizers"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|c| c["type"] == "Split")
            .map(|c| serde_json::from_value(c.clone()).unwrap())
            .collect();
        assert_eq!(splits.len(), 3, "deepseek has 3 Splits");
        let seq = Sequence::new(splits);
        assert!(seq.is_deepseek(), "deepseek's exact 3-Split sequence must be recognized");

        for text in [
            "中文 with 123 numbers!! and ケーキ don't",
            "hello 世界\n\n表 x",
            "純粋なCJK日本語テキスト",
            "  spaces  and\ttabs 42 café Naïve",
        ] {
            assert_eq!(
                pipeline_pretokenize(&seq, text),
                legacy_pretokenize(&seq, text),
                "deepseek diverged on {text:?}",
            );
        }
    }

    // CJK-range PUNCTUATION (・ U+30FB, ゠, ゛゜) sits inside Split-1's `[一-龥぀-ゟ゠-ヿ]` range, so
    // Split-1 isolates it (`fsm_deepseek` handles a CJK-range run as a closed unit) — a preceding space
    // stays separate and it never merges with adjacent non-CJK punct.
    #[test]
    fn pipeline_deepseek_cjk_punct_whitespace_edge() {
        let path = "../data/deepseek-v4-flash-base-tokenizer.json";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let splits: Vec<PreTokenizerWrapper> = v["pre_tokenizer"]["pretokenizers"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|c| c["type"] == "Split")
            .map(|c| serde_json::from_value(c.clone()).unwrap())
            .collect();
        let seq = Sequence::new(splits);
        let text = "hello 世界\n\n表 ・ x"; // standalone ・ with surrounding spaces
        assert_eq!(pipeline_pretokenize(&seq, text), legacy_pretokenize(&seq, text));
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
