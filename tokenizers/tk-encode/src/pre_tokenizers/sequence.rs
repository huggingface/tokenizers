use std::convert::{TryFrom, TryInto};

use crate::pipeline::{self, PipelinePreTokenizer};
use crate::pre_tokenizers::PreTokenizerWrapper;
use crate::tokenizer::Result;
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

#[derive(Clone, Debug, PartialEq)]
pub struct PipelineSequence {
    pre_tokenizers: Vec<PipelinePreTokenizer>,
}

impl PipelineSequence {
    pub fn new(pre_tokenizers: Vec<PipelinePreTokenizer>) -> Self {
        Self { pre_tokenizers }
    }

    /// Same recognition as [`crate::utils::is_deepseek`], on the converted children: the first three are
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

// SAFETY: a `Sequence` runs its children in order. A child splits the spans emitted by the previous child.
// `Sequence` is safe because:
// - all of its children are safe
// - offsets added by the sequence are correct and land on character boundaries
//
// The deepseek fast path has no children to run: it calls an `atomsplit` fsm, which splits only at
// character boundaries of `text`. See the `atomsplit::fsm` docs.
unsafe impl pipeline::PreTokenizer for PipelineSequence {
    /// Runs each child in turn, where every child subdivides the spans produced
    /// so far. A child sees only the text of a span (`&text[span]`) and returns
    /// offsets relative to it, which we rebase to absolute mirroring how the
    /// legacy path worked.
    fn pre_tokenize(
        &self,
        text: &str,
        scratch: &mut pipeline::PreTokenizerScratch,
        out: &mut Vec<pipeline::Span>,
    ) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }

        // deepseek's 3-Split composition → one native FSM pass (also lets the Sequence handle the
        // trailing byte-map ByteLevel, which the generic child loop can't range-split).
        if self.is_deepseek() {
            scratch.split_on_tags(text.as_bytes(), atomsplit::fsm::fsm_deepseek, out);
            return Ok(());
        }

        // Fuse Split+ByteLevel: a byte-map `ByteLevel` (use_regex=false) converts to
        // `PipelinePreTokenizer::None`, a pure identity pass. Skipping the `None`s collapses the
        // dominant `Sequence[Split(regex), ByteLevel]` archetype (~40% of Hub usage) to a lone child we
        // run straight into `out` — no double-buffer, no rebase, no redundant identity pass over every
        // token. Sequences with ≥2 real children (or none) fall through to the generic loop unchanged.
        let mut work = self
            .pre_tokenizers
            .iter()
            .filter(|c| !matches!(c, PipelinePreTokenizer::None));
        if let (Some(only), None) = (work.next(), work.next()) {
            return pipeline::PreTokenizer::pre_tokenize(only, text, scratch, out);
        }

        let [mut current, mut next] = scratch.take_pair();
        current.clear();
        current.push(pipeline::Span {
            start: 0,
            end: text.len() as u32,
        });

        for child in &self.pre_tokenizers {
            next.clear();
            for span in &current {
                let base = span.start;
                // The child appends span-relative spans straight into `next`;
                // rebase just those to absolute in place — no scratch buffer.
                let from = next.len();
                pipeline::PreTokenizer::pre_tokenize(
                    child,
                    &text[span.range()],
                    scratch,
                    &mut next,
                )?;
                // FIXME: do we want to add an `offset` param to `pre_tokenize` so we don't have to
                // rebase?
                for s in &mut next[from..] {
                    s.start += base;
                    s.end += base;
                }
            }
            std::mem::swap(&mut current, &mut next);
        }

        // Every call the pipeline makes arrives with `out` empty, since `encode_sequence` clears
        // `pre_tokens` before pre-tokenizing: hand it the buffer the loop just filled instead of
        // copying. `current` takes `out`'s allocation in exchange and goes back to the scratch,
        // and both are pooled, so which buffer ends up where does not matter.
        if out.is_empty() {
            std::mem::swap(out, &mut current);
        } else {
            out.extend_from_slice(&current);
        }
        scratch.put_pair([current, next]);
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

    /// Run the pipeline path and return `(piece, (start, end))` for each split.
    fn pipeline_pretokenize(seq: &PipelineSequence, text: &str) -> Vec<(String, (usize, usize))> {
        let mut scratch = pipeline::PreTokenizerScratch::default();
        let mut out = Vec::new();
        crate::pipeline::PreTokenizer::pre_tokenize(seq, text, &mut scratch, &mut out).unwrap();
        out.iter()
            .map(|s| {
                (
                    text[s.range()].to_string(),
                    (s.start as usize, s.end as usize),
                )
            })
            .collect()
    }

    /// `(piece, (start, end))` per split, frozen from the behaviour each pipeline shape
    /// was built to reproduce. Indexed `[config][text]`.
    type Golden = &'static [&'static [(&'static str, (usize, usize))]];

    fn assert_matches_golden(
        got: Vec<(String, (usize, usize))>,
        want: &[(&str, (usize, usize))],
        ctx: &str,
    ) {
        let got: Vec<(&str, (usize, usize))> = got.iter().map(|(s, o)| (s.as_str(), *o)).collect();
        assert_eq!(got.as_slice(), want, "{ctx}");
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
    fn pipeline_matches_golden() {
        // The pipeline path over varied configs (incl. a nested Sequence) and multi-script texts.
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
        assert_eq!(CONFIG_GOLDEN.len(), configs.len(), "one block per config");
        for (ci, cfg) in configs.into_iter().enumerate() {
            let pipe_seq: PipelineSequence = Sequence::new(cfg)
                .try_into()
                .expect("Failed to convert Sequence to PipelineSequence");
            assert_eq!(CONFIG_GOLDEN[ci].len(), texts.len(), "one row per text");
            for (ti, text) in texts.iter().enumerate() {
                assert_matches_golden(
                    pipeline_pretokenize(&pipe_seq, text),
                    CONFIG_GOLDEN[ci][ti],
                    &format!("config #{ci} on {text:?}"),
                );
            }
        }
    }

    const CONFIG_GOLDEN: &[Golden] = &[
        &[
            &[
                ("Hey", (0, 3)),
                ("friend", (4, 10)),
                ("!", (10, 11)),
                ("How", (16, 19)),
                ("are", (20, 23)),
                ("you", (24, 27)),
                ("?", (27, 28)),
                ("!", (28, 29)),
                ("?", (29, 30)),
            ],
            &[
                ("abc", (0, 3)),
                ("123", (4, 7)),
                ("def", (8, 11)),
                ("!", (11, 12)),
                ("!", (12, 13)),
                ("ghi", (13, 16)),
                ("42", (17, 19)),
            ],
            &[
                ("leading", (2, 9)),
                ("and", (11, 14)),
                ("trailing", (17, 25)),
                ("spaces", (26, 32)),
            ],
            &[
                ("café", (0, 5)),
                ("?", (5, 6)),
                ("no", (7, 9)),
                ("—", (9, 12)),
                ("maybe", (12, 17)),
                ("3", (18, 19)),
                (".", (19, 20)),
                ("14", (20, 22)),
                ("ok", (23, 25)),
            ],
            &[
                ("中文", (0, 6)),
                ("text", (7, 11)),
                ("123", (12, 15)),
                (",", (15, 16)),
                ("mixed", (17, 22)),
                ("!", (22, 23)),
            ],
            &[("single", (0, 6))],
            &[("!", (0, 1)), ("!", (1, 2)), ("!", (2, 3))],
        ],
        &[
            &[
                ("Hey", (0, 3)),
                ("friend", (4, 10)),
                ("!", (10, 11)),
                ("How", (16, 19)),
                ("are", (20, 23)),
                ("you", (24, 27)),
                ("?!?", (27, 30)),
            ],
            &[
                ("abc", (0, 3)),
                ("123", (4, 7)),
                ("def", (8, 11)),
                ("!!", (11, 13)),
                ("ghi", (13, 16)),
                ("42", (17, 19)),
            ],
            &[
                ("leading", (2, 9)),
                ("and", (11, 14)),
                ("trailing", (17, 25)),
                ("spaces", (26, 32)),
            ],
            &[
                ("café", (0, 5)),
                ("?", (5, 6)),
                ("no", (7, 9)),
                ("—", (9, 12)),
                ("maybe", (12, 17)),
                ("3", (18, 19)),
                (".", (19, 20)),
                ("14", (20, 22)),
                ("ok", (23, 25)),
            ],
            &[
                ("中文", (0, 6)),
                ("text", (7, 11)),
                ("123", (12, 15)),
                (",", (15, 16)),
                ("mixed", (17, 22)),
                ("!", (22, 23)),
            ],
            &[("single", (0, 6))],
            &[("!!!", (0, 3))],
        ],
        &[
            &[
                ("Hey", (0, 3)),
                ("friend", (4, 10)),
                ("!", (10, 11)),
                ("How", (16, 19)),
                ("are", (20, 23)),
                ("you", (24, 27)),
                ("?", (27, 28)),
                ("!", (28, 29)),
                ("?", (29, 30)),
            ],
            &[
                ("abc", (0, 3)),
                ("1", (4, 5)),
                ("2", (5, 6)),
                ("3", (6, 7)),
                ("def", (8, 11)),
                ("!", (11, 12)),
                ("!", (12, 13)),
                ("ghi", (13, 16)),
                ("4", (17, 18)),
                ("2", (18, 19)),
            ],
            &[
                ("leading", (2, 9)),
                ("and", (11, 14)),
                ("trailing", (17, 25)),
                ("spaces", (26, 32)),
            ],
            &[
                ("café", (0, 5)),
                ("?", (5, 6)),
                ("no", (7, 9)),
                ("—", (9, 12)),
                ("maybe", (12, 17)),
                ("3", (18, 19)),
                (".", (19, 20)),
                ("1", (20, 21)),
                ("4", (21, 22)),
                ("ok", (23, 25)),
            ],
            &[
                ("中文", (0, 6)),
                ("text", (7, 11)),
                ("1", (12, 13)),
                ("2", (13, 14)),
                ("3", (14, 15)),
                (",", (15, 16)),
                ("mixed", (17, 22)),
                ("!", (22, 23)),
            ],
            &[("single", (0, 6))],
            &[("!", (0, 1)), ("!", (1, 2)), ("!", (2, 3))],
        ],
        &[
            &[
                ("Hey", (0, 3)),
                ("friend", (4, 10)),
                ("!", (10, 11)),
                ("How", (16, 19)),
                ("are", (20, 23)),
                ("you", (24, 27)),
                ("?", (27, 28)),
                ("!", (28, 29)),
                ("?", (29, 30)),
            ],
            &[
                ("abc", (0, 3)),
                ("123", (4, 7)),
                ("def", (8, 11)),
                ("!", (11, 12)),
                ("!", (12, 13)),
                ("ghi", (13, 16)),
                ("42", (17, 19)),
            ],
            &[
                ("leading", (2, 9)),
                ("and", (11, 14)),
                ("trailing", (17, 25)),
                ("spaces", (26, 32)),
            ],
            &[
                ("café", (0, 5)),
                ("?", (5, 6)),
                ("no", (7, 9)),
                ("—", (9, 12)),
                ("maybe", (12, 17)),
                ("3", (18, 19)),
                (".", (19, 20)),
                ("14", (20, 22)),
                ("ok", (23, 25)),
            ],
            &[
                ("中文", (0, 6)),
                ("text", (7, 11)),
                ("123", (12, 15)),
                (",", (15, 16)),
                ("mixed", (17, 22)),
                ("!", (22, 23)),
            ],
            &[("single", (0, 6))],
            &[("!", (0, 1)), ("!", (1, 2)), ("!", (2, 3))],
        ],
    ];
    const BYTE_LEVEL_GOLDEN: Golden = &[
        &[
            ("Hello", (0, 5)),
            ("Ġthere", (5, 11)),
            ("Ċ", (11, 12)),
            ("Hello", (12, 17)),
            ("Ġthere", (17, 23)),
        ],
        &[
            ("ä¸Ńæĸĩ", (0, 6)),
            ("Ġtext", (6, 11)),
            ("Ġ123", (11, 15)),
            (",", (15, 16)),
            ("Ġmixed", (16, 22)),
            ("!", (22, 23)),
            ("ĠðŁ¤Ĺ", (23, 28)),
        ],
        &[
            ("I", (0, 1)),
            ("'m", (1, 3)),
            ("Ġsure", (3, 8)),
            ("Ġit", (8, 11)),
            ("'s", (11, 13)),
            ("Ġfine", (13, 18)),
            ("ĠĠĠ", (18, 21)),
        ],
    ];
    const DESERIALIZED_GOLDEN: &[(&str, (usize, usize))] = &[
        ("Hey", (0, 3)),
        ("friend!", (4, 11)),
        ("How", (16, 19)),
        ("are", (20, 23)),
        ("you?!?", (24, 30)),
    ];

    /// deepseek-v4's real `pre_tokenizer`: a `Sequence` of 3 Isolated `Split`s plus a trailing
    /// byte-map `ByteLevel`. Only the `Split`s are rebuilt here — the byte map is a separate step.
    #[cfg(feature = "fancy-regex")] // deepseek `Split`s need a backend at construction
    fn deepseek_splits() -> PipelineSequence {
        let path = "../data/deepseek-v4.json";
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
        Sequence::new(splits).try_into().unwrap()
    }

    #[cfg(feature = "fancy-regex")]
    #[test]
    fn pipeline_deepseek_uses_fsm_and_matches_golden() {
        let pipe = deepseek_splits();
        assert!(
            pipe.is_deepseek(),
            "deepseek's exact 3-Split sequence must be recognized"
        );

        assert_eq!(
            DEEPSEEK_GOLDEN.len(),
            DEEPSEEK_TEXTS.len(),
            "one row per text"
        );
        for (text, want) in DEEPSEEK_TEXTS.iter().zip(DEEPSEEK_GOLDEN) {
            assert_matches_golden(pipeline_pretokenize(&pipe, text), want, text);
        }
    }

    const DEEPSEEK_TEXTS: &[&str] = &[
        "中文 with 123 numbers!! and ケーキ don't",
        "hello 世界\n\n表 x",
        "純粋なCJK日本語テキスト",
        "  spaces  and\ttabs 42 café Naïve",
    ];

    const DEEPSEEK_GOLDEN: Golden = &[
        &[
            ("中文", (0, 6)),
            (" with", (6, 11)),
            (" ", (11, 12)),
            ("123", (12, 15)),
            (" numbers", (15, 23)),
            ("!!", (23, 25)),
            (" and", (25, 29)),
            (" ", (29, 30)),
            ("ケーキ", (30, 39)),
            (" don", (39, 43)),
            ("'t", (43, 45)),
        ],
        &[
            ("hello", (0, 5)),
            (" ", (5, 6)),
            ("世界", (6, 12)),
            ("\n\n", (12, 14)),
            ("表", (14, 17)),
            (" x", (17, 19)),
        ],
        &[
            ("純粋な", (0, 9)),
            ("CJK", (9, 12)),
            ("日本語テキスト", (12, 33)),
        ],
        &[
            (" ", (0, 1)),
            (" spaces", (1, 8)),
            (" ", (8, 9)),
            (" and", (9, 13)),
            ("\ttabs", (13, 18)),
            (" ", (18, 19)),
            ("42", (19, 21)),
            (" café", (21, 27)),
            (" Naïve", (27, 34)),
        ],
    ];

    // CJK-range PUNCTUATION (・ U+30FB, ゠, ゛゜) sits inside Split-1's `[一-龥぀-ゟ゠-ヿ]` range, so
    // Split-1 isolates it (`fsm_deepseek` handles a CJK-range run as a closed unit) — a preceding
    // space stays separate and it never merges with adjacent non-CJK punct.
    #[cfg(feature = "fancy-regex")]
    #[test]
    fn pipeline_deepseek_cjk_punct_whitespace_edge() {
        let text = "hello 世界\n\n表 ・ x"; // standalone ・ with surrounding spaces
        assert_matches_golden(
            pipeline_pretokenize(&deepseek_splits(), text),
            CJK_PUNCT_GOLDEN,
            text,
        );
    }

    const CJK_PUNCT_GOLDEN: &[(&str, (usize, usize))] = &[
        ("hello", (0, 5)),
        (" ", (5, 6)),
        ("世界", (6, 12)),
        ("\n\n", (12, 14)),
        ("表", (14, 17)),
        (" ", (17, 18)),
        ("・", (18, 21)),
        (" x", (21, 23)),
    ];

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
        use crate::SplitDelimiterBehavior;
        use crate::pre_tokenizers::split::{Split, SplitPattern};
        use crate::utils::byte_level::BYTES_CHAR_LOOKUP;
        use crate::utils::byte_level::GPT2_REGEX_STR;

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
        const TEXTS: &[&str] = &[
            "Hello there\nHello there",
            "中文 text 123, mixed! 🤗",
            "I'm sure it's fine   ",
        ];
        assert_eq!(BYTE_LEVEL_GOLDEN.len(), TEXTS.len(), "one row per text");
        for (text, want) in TEXTS.iter().zip(BYTE_LEVEL_GOLDEN) {
            let mut scratch = pipeline::PreTokenizerScratch::default();
            let mut out = Vec::new();
            crate::pipeline::PreTokenizer::pre_tokenize(&pipe_seq, text, &mut scratch, &mut out)
                .unwrap();
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
            assert_matches_golden(pipeline, want, text);
        }
    }

    #[test]
    fn deserialized_sequence_matches_golden() {
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
        assert_matches_golden(
            pipeline_pretokenize(&pipe_seq, text),
            DESERIALIZED_GOLDEN,
            text,
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
}
