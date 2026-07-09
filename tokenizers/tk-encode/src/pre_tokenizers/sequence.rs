use std::convert::{TryFrom, TryInto};

use crate::pipeline::{self, PipelinePreTokenizer};
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

#[derive(Clone, Debug, PartialEq)]
pub struct PipelineSequence {
    pre_tokenizers: Vec<PipelinePreTokenizer>,
}

impl PipelineSequence {
    pub fn new(pre_tokenizers: Vec<PipelinePreTokenizer>) -> Self {
        Self { pre_tokenizers }
    }

    /// Same recognition as [`Sequence::is_deepseek`], on the converted children: the first three are
    /// Isolated, non-inverted `Split`s carrying deepseek's `[\p{N}{1,3}, CJK, big]` regexes (the trailing
    /// byte-map `ByteLevel` converts to `PipelinePreTokenizer::None`). Routes the whole split to one
    /// `fsm_deepseek` pass.
    fn is_deepseek(&self) -> bool {
        use crate::pre_tokenizers::split::SplitPattern;
        use crate::tokenizer::SplitDelimiterBehavior::Isolated;
        let regex = |i: usize| match self.pre_tokenizers.get(i) {
            Some(PipelinePreTokenizer::Split(s)) if s.behavior == Isolated && !s.invert => {
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

impl TryFrom<Sequence> for PipelineSequence {
    type Error = crate::Error;
    fn try_from(value: Sequence) -> Result<Self> {
        Ok(Self {
            pre_tokenizers: value
                .pretokenizers
                .into_iter()
                .map(TryInto::try_into)
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

impl pipeline::PreTokenizer for PipelineSequence {
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
            out.extend(
                spans[..n]
                    .iter()
                    .map(|&(s, e)| pipeline::Split { start: s, end: e }),
            );
            return Ok(());
        }

        let cap = text.len() / 5;

        let mut current: Vec<pipeline::Split> = Vec::with_capacity(cap);
        current.push(pipeline::Split {
            start: 0,
            end: text.len() as u32,
        });
        let mut next: Vec<pipeline::Split> = Vec::with_capacity(cap);

        for child in &self.pre_tokenizers {
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
    fn pipeline_pretokenize(seq: &PipelineSequence, text: &str) -> Vec<(String, (usize, usize))> {
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
        let pipe_seq = seq
            .clone()
            .try_into()
            .expect("Failed to convert Sequence to PipelineSequence");
        assert_eq!(
            pipeline_pretokenize(&pipe_seq, "Hey friend!     How are you?!?"),
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
            let pipe_seq = seq
                .clone()
                .try_into()
                .expect("Failed to convert Sequence to PipelineSequence");
            for text in texts {
                assert_eq!(
                    pipeline_pretokenize(&pipe_seq, text),
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
        let pipe: PipelineSequence = seq.clone().try_into().unwrap();
        assert!(
            pipe.is_deepseek(),
            "deepseek's exact 3-Split sequence must be recognized"
        );

        for text in [
            "中文 with 123 numbers!! and ケーキ don't",
            "hello 世界\n\n表 x",
            "純粋なCJK日本語テキスト",
            "  spaces  and\ttabs 42 café Naïve",
        ] {
            assert_eq!(
                pipeline_pretokenize(&pipe, text),
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
        let pipe: PipelineSequence = seq.clone().try_into().unwrap();
        let text = "hello 世界\n\n表 ・ x"; // standalone ・ with surrounding spaces
        assert_eq!(
            pipeline_pretokenize(&pipe, text),
            legacy_pretokenize(&seq, text)
        );
    }

    // fsm_deepseek == the 3-Split onig Sequence over multilingual Wikipedia corpora — the broad byte-exact
    // guard. `he.txt` is why it exists: Hebrew mixes format controls (RLM, `\p{Cf}`) and Other_Alphabetic
    // symbols (Ⓘ, `\p{S}` but is_alphabetic), which stress the *gap* grouping (consecutive unmatched chars
    // = one piece) and the `ALPHA_SYM` Mark refinement (a `\w` char that is NOT `[\p{L}\p{M}]`).
    #[test]
    fn pipeline_deepseek_matches_legacy_on_corpora() {
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
        let pipe: PipelineSequence = seq.clone().try_into().unwrap();
        assert!(pipe.is_deepseek());
        // `he`/`ar` (RTL, RLM/format-mark + Other_Alphabetic-symbol heavy) are the cases the atomsplit
        // deepseek bench doesn't cover; the other 8 languages are byte-exact-gated there.
        for lang in ["he", "ar"] {
            let Ok(corpus) =
                std::fs::read_to_string(format!("../atomsplit/benches/data/{lang}.txt"))
            else {
                continue;
            };
            for (ln, line) in corpus.lines().enumerate() {
                if line.is_empty() {
                    continue;
                }
                let (p, l) = (
                    pipeline_pretokenize(&pipe, line),
                    legacy_pretokenize(&seq, line),
                );
                if p != l {
                    let k = p
                        .iter()
                        .zip(l.iter())
                        .position(|(a, b)| a != b)
                        .unwrap_or(p.len().min(l.len()));
                    let lo = k.saturating_sub(1);
                    panic!(
                        "deepseek diverged {lang}.txt:{ln} @tok {k}\n  {line:?}\n  pipe: {:?}\n  legc: {:?}",
                        &p[lo..(k + 3).min(p.len())],
                        &l[lo..(k + 3).min(l.len())],
                    );
                }
            }
        }
    }

    #[test]
    fn pipeline_empty_input() {
        let seq = Sequence::new(vec![
            PreTokenizerWrapper::WhitespaceSplit(WhitespaceSplit),
            PreTokenizerWrapper::Punctuation(Punctuation::default()),
        ]);
        let pipe_seq = seq
            .clone()
            .try_into()
            .expect("Failed to convert Sequence to PipelineSequence");

        assert!(pipeline_pretokenize(&pipe_seq, "").is_empty());
    }

    #[test]
    fn pipeline_matches_legacy_oracle_byte_level() {
        // The Llama-3 / DeepSeek archetype: Sequence[Split(regex), ByteLevel(use_regex=false)].
        // Pipeline ranges must match the legacy oracle's Original-referential offsets, and the
        // byte-level transform of each range must match the legacy split string.
        use crate::pre_tokenizers::split::{Split, SplitPattern};
        use crate::utils::byte_level::BYTES_CHAR_LOOKUP;
        use crate::utils::byte_level::GPT2_REGEX_STR;
        use crate::SplitDelimiterBehavior;

        let seq = Sequence::new(vec![
            PreTokenizerWrapper::Split(
                Split::new(
                    SplitPattern::Regex(GPT2_REGEX_STR.to_owned()),
                    SplitDelimiterBehavior::Isolated,
                    false,
                )
                .unwrap(),
            ),
            PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, true, false)),
        ]);
        let pipe_seq: PipelineSequence = seq
            .clone()
            .try_into()
            .expect("Failed to convert Sequence to PipelineSequence");
        for text in [
            "Hello there\nHello there",
            "中文 text 123, mixed! 🤗",
            "I'm sure it's fine   ",
        ] {
            let mut out = Vec::new();
            crate::pipeline::PreTokenizer::pre_tokenize(&pipe_seq, text, &mut out).unwrap();
            let pipeline: Vec<(String, (usize, usize))> = out
                .iter()
                .map(|s| {
                    let transformed = text[s.range()]
                        .bytes()
                        .map(|b| BYTES_CHAR_LOOKUP[b as usize])
                        .collect();
                    (transformed, (s.start as usize, s.end as usize))
                })
                .collect();
            assert_eq!(
                pipeline,
                legacy_pretokenize(&seq, text),
                "diverged on {text:?}"
            );
        }
    }

    #[test]
    fn deserialized_sequence_matches_legacy_oracle() {
        // Real tokenizers are loaded via serde, not `Sequence::new` — the pipeline
        // path must behave identically for a deserialized Sequence.
        let seq: Sequence = serde_json::from_str(
            r#"{"type":"Sequence","pretokenizers":[{"type":"WhitespaceSplit"}]}"#,
        )
        .unwrap();
        let pipe_seq = seq
            .clone()
            .try_into()
            .expect("Failed to convert Sequence to PipelineSequence");

        let text = "Hey friend!     How are you?!?";
        assert_eq!(
            pipeline_pretokenize(&pipe_seq, text),
            legacy_pretokenize(&seq, text),
        );
    }

    #[test]
    fn pipeline_unsupported_child_errors() {
        // Metaspace has no range-based form. Constructing the Sequence must still work
        // (the legacy path supports it) — only the pipeline conversion should fail.
        use std::convert::TryFrom;
        let seq = Sequence::new(vec![
            PreTokenizerWrapper::WhitespaceSplit(WhitespaceSplit),
            PreTokenizerWrapper::Metaspace(crate::pre_tokenizers::metaspace::Metaspace::default()),
        ]);
        assert!(PipelinePreTokenizer::try_from(PreTokenizerWrapper::Sequence(seq)).is_err());
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
