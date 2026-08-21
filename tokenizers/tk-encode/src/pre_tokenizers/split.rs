use crate::pipeline;
use crate::utils::{GptFsm, GptFsmPattern, SysRegex, gpt_fsm};
use atomsplit::literal::Literal;
use serde::{Deserialize, Deserializer, Serialize};

use crate::tokenizer::{
    PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior,
    pattern::{Invert, Pattern},
};

/// Represents the different patterns that `Split` can use
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Eq)]
pub enum SplitPattern {
    String(String),
    Regex(String),
}

impl From<String> for SplitPattern {
    fn from(v: String) -> Self {
        Self::String(v)
    }
}

impl From<&str> for SplitPattern {
    fn from(v: &str) -> Self {
        Self::String(v.to_owned())
    }
}

/// How a [`Split`] looks for its pattern.
#[derive(Debug)]
pub enum Search {
    /// A plain string, scanned for directly — no regex engine involved, so this works in every build.
    Literal(Literal),
    /// A real regex, which needs the system backend (the `fancy-regex` feature).
    Regex(SysRegex),
    /// No way to search: no backend is compiled and the pattern is a regex. Splitting then only works
    /// through the native FSM below, and errors otherwise.
    Unavailable,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub struct Split {
    pub pattern: SplitPattern,
    /// How the pattern is found. A plain string never needs a backend; a regex does, unless it is one
    /// of the GPT patterns the native FSM below covers.
    #[serde(skip)]
    pub search: Search,
    pub behavior: SplitDelimiterBehavior,
    pub invert: bool,
    /// Native `atomsplit` FSM for a recognized GPT regex (gpt2 / cl100k-Llama-3 / o200k), used on the
    /// pipeline path when `behavior == Isolated && !invert` (how these regexes always ship). Byte-exact
    /// with `regex`; `None` falls back to `regex`.
    #[serde(skip)]
    fsm: Option<GptFsm>,
}

impl<'de> Deserialize<'de> for Split {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        enum Type {
            Split,
        }

        #[derive(Deserialize)]
        pub struct SplitHelper {
            #[serde(rename = "type")]
            _type: Type,
            pattern: SplitPattern,
            behavior: SplitDelimiterBehavior,
            invert: bool,
        }

        let helper = SplitHelper::deserialize(deserializer)?;
        Self::new(helper.pattern, helper.behavior, helper.invert).map_err(serde::de::Error::custom)
    }
}

impl Clone for Split {
    fn clone(&self) -> Self {
        Self::new(self.pattern.clone(), self.behavior, self.invert).unwrap()
    }
}

impl PartialEq for Split {
    fn eq(&self, other: &Self) -> bool {
        self.pattern == other.pattern
            && self.behavior == other.behavior
            && self.invert == other.invert
    }
}

impl Split {
    pub fn new<I: Into<SplitPattern>>(
        pattern: I,
        behavior: SplitDelimiterBehavior,
        invert: bool,
    ) -> Result<Self> {
        let pattern: SplitPattern = pattern.into();
        let fsm = match &pattern {
            SplitPattern::String(_) => None,
            SplitPattern::Regex(r) => gpt_fsm(r),
        };
        let search = match &pattern {
            SplitPattern::String(s) => Search::Literal(Literal::new(s.as_bytes())?),
            // A regex needs the system backend. Missing it is only fatal when the native FSM cannot
            // cover this pattern either — which `pre_tokenize` reports, since a recognized GPT
            // pattern in its usual form splits without any backend.
            SplitPattern::Regex(r) => match SysRegex::new(r) {
                Ok(regex) => Search::Regex(regex),
                Err(_) if fsm.is_some() => Search::Unavailable,
                Err(e) => return Err(e),
            },
        };

        Ok(Self {
            pattern,
            search,
            behavior,
            invert,
            fsm,
        })
    }

    /// A `Split` that is known to be driven natively, so no regex backend is compiled.
    ///
    /// [`Split::new`] asks the system regex to compile every regex pattern, which a build without
    /// `fancy-regex` has no engine for. Two cases do not need one: a pattern [`gpt_fsm`] recognises,
    /// and a member of a composition the pipeline runs as a single native pass -- deepseek's three
    /// regexes are individually unrecognised but never individually run, so `Split::new` would
    /// reject them. A literal pattern is searched for directly and never needed an engine either.
    pub fn native(
        pattern: SplitPattern,
        behavior: SplitDelimiterBehavior,
        invert: bool,
    ) -> Result<Self> {
        let fsm = match &pattern {
            SplitPattern::String(_) => None,
            SplitPattern::Regex(r) => gpt_fsm(r),
        };
        let search = match &pattern {
            SplitPattern::String(s) => Search::Literal(Literal::new(s.as_bytes())?),
            SplitPattern::Regex(_) => Search::Unavailable,
        };
        Ok(Self {
            pattern,
            search,
            behavior,
            invert,
            fsm,
        })
    }

    /// Pipeline canonicalization. A recognized whole-covering GPT regex shipped
    /// as `(invert=true, behavior=Removed)` — the tiktoken-conversion convention
    /// used by cl100k/o200k — is byte-exactly equivalent to `(invert=false,
    /// Isolated)`, the form the native FSM fast path requires (the inverted match
    /// set is the gaps, and these patterns leave no gaps). Rewrite to it so
    /// cl100k/o200k route to `fsm_cl100k`/`fsm_o200k` instead of the SysRegex fallback.
    // `pub` because the `PreTokenizerWrapper` -> `PipelinePreTokenizer` lowering, which is the only
    // caller, lives in `tk-convert`.
    pub fn canonicalized_for_pipeline(self) -> Result<Self> {
        use crate::tokenizer::SplitDelimiterBehavior::{Isolated, Removed};
        if self.fsm.is_some() && self.invert && self.behavior == Removed {
            Split::new(self.pattern, Isolated, false)
        } else {
            Ok(self)
        }
    }
}

impl PreTokenizer for Split {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        match &self.search {
            Search::Literal(literal) => {
                return if self.invert {
                    pretokenized
                        .split(|_, normalized| normalized.split(Invert(literal), self.behavior))
                } else {
                    pretokenized.split(|_, normalized| normalized.split(literal, self.behavior))
                };
            }
            Search::Regex(regex) => {
                return if self.invert {
                    pretokenized
                        .split(|_, normalized| normalized.split(Invert(regex), self.behavior))
                } else {
                    pretokenized.split(|_, normalized| normalized.split(regex, self.behavior))
                };
            }
            Search::Unavailable => {}
        }
        // No way to search for the pattern: only a recognized GPT pattern in its canonical usage
        // (Isolated, not inverted — how these regexes always ship) can split, via the native FSM.
        let fsm = self
            .fsm
            .filter(|_| !self.invert && self.behavior == SplitDelimiterBehavior::Isolated)
            .ok_or_else(|| -> crate::tokenizer::Error {
                "this `Split` pattern needs a system-regex backend; enable the `fancy-regex` feature"
                    .into()
            })?;
        pretokenized.split(|_, normalized| {
            normalized.split(GptFsmPattern(fsm), SplitDelimiterBehavior::Isolated)
        })
    }
}

// SAFETY: both routes cut only at character boundaries of `text`. The native route is an `atomsplit`
// fsm, see "What the spans guarantee" in its docs. The search route forwards the offsets of
// `Pattern::find_matches` through `pipeline::split_matches`, and every `Pattern` here reports
// boundaries: a regex matches on a `&str`, and `Literal` holds the bytes of a `&str` pattern, which
// well-formed UTF-8 can only contain starting and ending on a boundary.
unsafe impl pipeline::PreTokenizer for Split {
    fn pre_tokenize(
        &self,
        text: &str,
        scratch: &mut pipeline::PreTokenizerScratch,
        out: &mut Vec<pipeline::Span>,
    ) -> Result<()> {
        // A recognized GPT regex (gpt2 / cl100k-Llama-3) in its only real usage — `Isolated`, not
        // inverted — routes straight to the native atomsplit FSM. These regexes cover the whole input,
        // so `Isolated` == the match list, and the FSM is byte-exact with `regex` (see the tests).
        if let Some(fsm) = self
            .fsm
            .filter(|_| !self.invert && self.behavior == SplitDelimiterBehavior::Isolated)
        {
            scratch.split_on_tags(
                text.as_bytes(),
                |bytes, tags, spans| match fsm {
                    GptFsm::Cl100k { digit_cap } => {
                        atomsplit::fsm::fsm_cl100k_cap(bytes, tags, spans, digit_cap)
                    }
                    GptFsm::Gpt2 => atomsplit::fsm::fsm_byte_level(bytes, tags, spans),
                    GptFsm::O200k => atomsplit::fsm::fsm_o200k(bytes, tags, spans),
                    GptFsm::Tekken => atomsplit::fsm::fsm_tekken(bytes, tags, spans),
                },
                out,
            );
            return Ok(());
        }
        // Not a natively-routed GPT regex: fall-back to Literal or Regex search
        let matches = match (&self.search, self.invert) {
            (Search::Literal(literal), false) => literal.find_matches(text)?,
            (Search::Literal(literal), true) => Invert(literal).find_matches(text)?,
            (Search::Regex(regex), false) => regex.find_matches(text)?,
            (Search::Regex(regex), true) => Invert(regex).find_matches(text)?,
            (Search::Unavailable, _) => {
                return Err(
                    "this `Split` pattern needs a system-regex backend; enable the `fancy-regex` feature"
                        .into(),
                );
            }
        };
        pipeline::split_matches(out, matches, self.behavior);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OffsetReferential, OffsetType, PreTokenizer};
    use SplitDelimiterBehavior::*;

    #[cfg(feature = "fancy-regex")] // needs a system-regex backend
    #[test]
    fn basic() {
        let tests = vec![
            (
                Removed,
                "How are you doing?",
                vec![
                    ("How", (0, 3)),
                    ("are", (4, 7)),
                    ("you", (8, 11)),
                    ("doing", (12, 17)),
                    ("?", (17, 18)),
                ],
            ),
            (
                Isolated,
                "How are you doing?",
                vec![
                    ("How", (0, 3)),
                    (" ", (3, 4)),
                    ("are", (4, 7)),
                    (" ", (7, 8)),
                    ("you", (8, 11)),
                    (" ", (11, 12)),
                    ("doing", (12, 17)),
                    ("?", (17, 18)),
                ],
            ),
            (
                MergedWithPrevious,
                "How are you doing?",
                vec![
                    ("How ", (0, 4)),
                    ("are ", (4, 8)),
                    ("you ", (8, 12)),
                    ("doing", (12, 17)),
                    ("?", (17, 18)),
                ],
            ),
            (
                MergedWithNext,
                "How are you doing?",
                vec![
                    ("How", (0, 3)),
                    (" are", (3, 7)),
                    (" you", (7, 11)),
                    (" doing", (11, 17)),
                    ("?", (17, 18)),
                ],
            ),
            (
                Contiguous,
                "How are you doing?",
                vec![
                    ("How", (0, 3)),
                    (" ", (3, 4)),
                    ("are", (4, 7)),
                    (" ", (7, 8)),
                    ("you", (8, 11)),
                    (" ", (11, 12)),
                    ("doing?", (12, 18)),
                ],
            ),
        ];

        // use whitespace regex
        let regex = SplitPattern::Regex(r"\w+|[^\w\s]+".into());

        for (behavior, s, res) in tests {
            let mut pretokenized = PreTokenizedString::from(s);
            let pretok = Split::new(regex.clone(), behavior, true).unwrap();
            pretok.pre_tokenize(&mut pretokenized).unwrap();
            assert_eq!(
                pretokenized
                    .get_splits(OffsetReferential::Original, OffsetType::Byte)
                    .into_iter()
                    .map(|(s, o, _)| (s, o))
                    .collect::<Vec<_>>(),
                res
            );
        }
    }

    fn pipeline_split(
        pattern: SplitPattern,
        behavior: SplitDelimiterBehavior,
        invert: bool,
        text: &str,
    ) -> Vec<(&str, (u32, u32))> {
        let pretok = Split::new(pattern, behavior, invert).unwrap();
        let mut scratch = pipeline::PreTokenizerScratch::default();
        let mut splits = Vec::with_capacity(text.len() / 5);
        crate::pipeline::PreTokenizer::pre_tokenize(&pretok, text, &mut scratch, &mut splits)
            .unwrap();
        splits
            .iter()
            .map(|s| (&text[s.range()], (s.start, s.end)))
            .collect()
    }

    #[cfg(feature = "fancy-regex")] // needs a system-regex backend
    #[test]
    fn pipeline_matches_legacy() {
        let regex = SplitPattern::Regex(r"\w+|[^\w\s]+".into());
        #[allow(clippy::type_complexity)]
        let cases: Vec<(SplitDelimiterBehavior, Vec<(&str, (u32, u32))>)> = vec![
            (
                Removed,
                vec![
                    ("How", (0, 3)),
                    ("are", (4, 7)),
                    ("you", (8, 11)),
                    ("doing", (12, 17)),
                    ("?", (17, 18)),
                ],
            ),
            (
                Isolated,
                vec![
                    ("How", (0, 3)),
                    (" ", (3, 4)),
                    ("are", (4, 7)),
                    (" ", (7, 8)),
                    ("you", (8, 11)),
                    (" ", (11, 12)),
                    ("doing", (12, 17)),
                    ("?", (17, 18)),
                ],
            ),
            (
                MergedWithPrevious,
                vec![
                    ("How ", (0, 4)),
                    ("are ", (4, 8)),
                    ("you ", (8, 12)),
                    ("doing", (12, 17)),
                    ("?", (17, 18)),
                ],
            ),
            (
                MergedWithNext,
                vec![
                    ("How", (0, 3)),
                    (" are", (3, 7)),
                    (" you", (7, 11)),
                    (" doing", (11, 17)),
                    ("?", (17, 18)),
                ],
            ),
            (
                Contiguous,
                vec![
                    ("How", (0, 3)),
                    (" ", (3, 4)),
                    ("are", (4, 7)),
                    (" ", (7, 8)),
                    ("you", (8, 11)),
                    (" ", (11, 12)),
                    ("doing?", (12, 18)),
                ],
            ),
        ];
        for (behavior, expected) in cases {
            assert_eq!(
                pipeline_split(regex.clone(), behavior, true, "How are you doing?"),
                expected,
                "behavior: {behavior:?}",
            );
        }
    }

    #[cfg(feature = "fancy-regex")] // needs a system-regex backend
    #[test]
    fn pipeline_invert_and_edges() {
        // invert = false: split *on* the regex (whitespace), Removed drops it
        assert_eq!(
            pipeline_split(SplitPattern::Regex(r"\s+".into()), Removed, false, "a b  c"),
            vec![("a", (0, 1)), ("b", (2, 3)), ("c", (5, 6))],
        );
        // empty input
        assert_eq!(
            pipeline_split(SplitPattern::Regex(r"\s+".into()), Removed, false, ""),
            Vec::<(&str, (u32, u32))>::new(),
        );
        // string pattern (escaped literal), Isolated keeps the delimiter
        assert_eq!(
            pipeline_split("-".into(), Isolated, false, "a-b"),
            vec![("a", (0, 1)), ("-", (1, 2)), ("b", (2, 3))],
        );
    }

    #[test]
    fn pipeline_gpt2_uses_fsm_and_matches_legacy() {
        // The gpt2 pattern is recognized -> the pipeline path routes to the native
        // atomsplit FSM; its output must equal the legacy fancy-regex path.
        let gpt2 = atomsplit::regexes::GPT2;
        let corpus = "The quick brown fox 123!!!  double  spaces\tand tabs. don't Naïve café. ";
        let pretok = Split::new(SplitPattern::Regex(gpt2.into()), Isolated, false).unwrap();
        assert!(pretok.fsm.is_some(), "gpt2 pattern should be recognized");

        // legacy reference
        let mut pre = PreTokenizedString::from(corpus);
        pretok.pre_tokenize(&mut pre).unwrap();
        let legacy: Vec<(&str, (u32, u32))> = pre
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(s, o, _)| (s, (o.0 as u32, o.1 as u32)))
            .collect();

        assert_eq!(
            pipeline_split(SplitPattern::Regex(gpt2.into()), Isolated, false, corpus),
            legacy,
        );
    }

    #[test]
    fn pipeline_cl100k_llama3_uses_fsm_and_matches_legacy() {
        // Llama-3's EXACT pre_tokenizer regex (from data/llama-3-tokenizer.json) → recognized → routes
        // to fsm_cl100k. Output must equal the legacy SysRegex Isolated split, byte-for-byte.
        let cl100k = atomsplit::regexes::CL100K;
        let corpus =
            "The quick brown fox 123!!!  double  spaces\tand tabs. don't Naïve café.\n\n世界 안녕 ";
        let pretok = Split::new(SplitPattern::Regex(cl100k.into()), Isolated, false).unwrap();
        assert!(
            pretok.fsm.is_some(),
            "cl100k / Llama-3 pattern should route to the native FSM"
        );

        let mut pre = PreTokenizedString::from(corpus);
        pretok.pre_tokenize(&mut pre).unwrap();
        let legacy: Vec<(&str, (u32, u32))> = pre
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(s, o, _)| (s, (o.0 as u32, o.1 as u32)))
            .collect();

        assert_eq!(
            pipeline_split(SplitPattern::Regex(cl100k.into()), Isolated, false, corpus),
            legacy,
        );
    }

    #[test]
    fn pipeline_o200k_uses_fsm_and_matches_legacy() {
        // GPT-4o's EXACT pre_tokenization regex → recognized → routes to fsm_o200k. Output must equal the
        // legacy SysRegex Isolated split, byte-for-byte. Corpus stresses the case-aware letter split:
        // camelCase, ALLCAPS→word, McDonald's-style, contractions, accented Ll (é/ß), CJK (caseless).
        let o200k = atomsplit::regexes::O200K;
        let corpus = "McDonald's iPhone SQLite HELLOWorld camelCase don't I'll We've 3.14 café Straße 世界 안녕\n\n  Mixed CASE end.";
        let pretok = Split::new(SplitPattern::Regex(o200k.into()), Isolated, false).unwrap();
        assert!(
            pretok.fsm.is_some(),
            "o200k / GPT-4o pattern should route to the native FSM"
        );

        let mut pre = PreTokenizedString::from(corpus);
        pretok.pre_tokenize(&mut pre).unwrap();
        let legacy: Vec<(&str, (u32, u32))> = pre
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(s, o, _)| (s, (o.0 as u32, o.1 as u32)))
            .collect();

        assert_eq!(
            pipeline_split(SplitPattern::Regex(o200k.into()), Isolated, false, corpus),
            legacy,
        );
    }

    #[test]
    fn pipeline_tekken_uses_fsm_and_matches_legacy() {
        // Mistral's tekken regex (mistral-small-4) → recognized → routes to fsm_tekken. Same corpus
        // shape as o200k, whose grammar it shares: the differences it must get right are apostrophes
        // (no contraction suffix, so `'s` starts a new token) and one token per digit.
        let tekken = atomsplit::regexes::TEKKEN;
        let corpus = "McDonald's iPhone SQLite HELLOWorld camelCase don't I'll We've 3.14159 café Straße 世界 안녕\n\n  path/to/file Mixed CASE end.";
        let pretok = Split::new(SplitPattern::Regex(tekken.into()), Isolated, false).unwrap();
        assert_eq!(
            pretok.fsm,
            Some(crate::utils::GptFsm::Tekken),
            "tekken / mistral pattern should route to the native FSM"
        );

        let mut pre = PreTokenizedString::from(corpus);
        pretok.pre_tokenize(&mut pre).unwrap();
        let legacy: Vec<(&str, (u32, u32))> = pre
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(s, o, _)| (s, (o.0 as u32, o.1 as u32)))
            .collect();

        assert_eq!(
            pipeline_split(SplitPattern::Regex(tekken.into()), Isolated, false, corpus),
            legacy,
        );
    }

    #[test]
    fn pipeline_qwen2_uses_fsm_and_matches_legacy() {
        // Qwen2's regex is cl100k character-for-character EXCEPT rule 3 is `\p{N}` (each digit its own
        // token) instead of `\p{N}{1,3}`. The structural recognizer extracts digit_cap=1 → fsm_cl100k_cap,
        // so it unrolls (no per-tokenizer exact-string entry). Corpus stresses multi-digit runs.
        let qwen2 = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        let corpus = "abc 12345 don't A1B2 3.14 100%%  double  spaces\ttab.\n\n世界 42";
        let pretok = Split::new(SplitPattern::Regex(qwen2.into()), Isolated, false).unwrap();
        assert_eq!(
            pretok.fsm,
            Some(crate::utils::GptFsm::Cl100k { digit_cap: 1 }),
            "Qwen2 pattern should route to the cl100k FSM with digit cap 1"
        );

        let mut pre = PreTokenizedString::from(corpus);
        pretok.pre_tokenize(&mut pre).unwrap();
        let legacy: Vec<(&str, (u32, u32))> = pre
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(s, o, _)| (s, (o.0 as u32, o.1 as u32)))
            .collect();

        assert_eq!(
            pipeline_split(SplitPattern::Regex(qwen2.into()), Isolated, false, corpus),
            legacy,
        );
    }

    /// The goal of the literal path: a plain string pattern splits with no regex backend, on both the
    /// legacy and the pipeline path.
    #[test]
    fn a_string_pattern_needs_no_backend() {
        let pretok = Split::new("-", SplitDelimiterBehavior::Removed, false).unwrap();
        let mut legacy = PreTokenizedString::from("a-b--c");
        pretok.pre_tokenize(&mut legacy).unwrap();
        let words: Vec<&str> = legacy
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .iter()
            .map(|(word, _, _)| *word)
            .collect();
        assert_eq!(words, ["a", "b", "c"]);
        assert_eq!(
            pipeline_split(
                SplitPattern::String("-".into()),
                SplitDelimiterBehavior::Removed,
                false,
                "a-b--c"
            ),
            [("a", (0, 1)), ("b", (2, 3)), ("c", (5, 6))]
        );
    }

    /// A config spelling its pattern as a string must also *deserialize* with no backend — the
    /// regex half of `serialization` below can only run once one is compiled.
    #[test]
    fn a_string_pattern_deserializes_with_no_backend() {
        let split_s =
            r#"{"type":"Split","pattern":{"String":"Hello"},"behavior":"Removed","invert":true}"#;
        let split = Split::new("Hello", SplitDelimiterBehavior::Removed, true).unwrap();
        assert_eq!(serde_json::from_str::<Split>(split_s).unwrap(), split);
        assert_eq!(serde_json::to_string(&split).unwrap(), split_s);
    }

    #[cfg(feature = "fancy-regex")] // needs a system-regex backend
    #[test]
    fn regex_string() {
        let mut pretok_str_for_regex = PreTokenizedString::from("Hey, man!");
        let mut pretok_str_for_string = pretok_str_for_regex.clone();

        // pre-tokenizer splits on " " - one from Regex, one from string
        let pretokenizer_regex = Split::new(
            SplitPattern::Regex(r"\s+".into()),
            SplitDelimiterBehavior::Removed,
            false,
        )
        .unwrap();
        let pretokenizer_string = Split::new(" ", SplitDelimiterBehavior::Removed, false).unwrap();

        pretokenizer_regex
            .pre_tokenize(&mut pretok_str_for_regex)
            .unwrap();
        pretokenizer_string
            .pre_tokenize(&mut pretok_str_for_string)
            .unwrap();

        assert_eq!(pretok_str_for_regex, pretok_str_for_string);
    }

    #[test]
    fn invert() {
        let mut pretok_str = PreTokenizedString::from("Hello Hello Hello");
        let mut pretok_str_for_invert = pretok_str.clone();

        // one pre-tokenizer splits on " " - one splits inverted on "Hello"
        let pretokenizer = Split::new(" ", SplitDelimiterBehavior::Removed, false).unwrap();
        let pretokenizer_invert =
            Split::new("Hello", SplitDelimiterBehavior::Removed, true).unwrap();

        pretokenizer.pre_tokenize(&mut pretok_str).unwrap();
        pretokenizer_invert
            .pre_tokenize(&mut pretok_str_for_invert)
            .unwrap();

        assert_eq!(pretok_str, pretok_str_for_invert);
    }

    #[cfg(feature = "fancy-regex")] // needs a system-regex backend
    #[test]
    fn serialization() {
        use SplitDelimiterBehavior::*;

        let split = Split::new("Hello", Removed, true).unwrap();
        let split_s =
            r#"{"type":"Split","pattern":{"String":"Hello"},"behavior":"Removed","invert":true}"#;
        assert_eq!(serde_json::to_string(&split).unwrap(), split_s);
        assert_eq!(serde_json::from_str::<Split>(split_s).unwrap(), split);

        let split = Split::new(SplitPattern::Regex(r"\s+".into()), Isolated, false).unwrap();
        let split_s =
            r#"{"type":"Split","pattern":{"Regex":"\\s+"},"behavior":"Isolated","invert":false}"#;
        assert_eq!(serde_json::to_string(&split).unwrap(), split_s);
        assert_eq!(serde_json::from_str::<Split>(split_s).unwrap(), split);
    }
}
