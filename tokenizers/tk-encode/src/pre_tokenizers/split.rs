use crate::pipeline;
use crate::utils::{GptFsm, SysRegex, gpt_fsm};
use atomsplit::literal::Literal;
use serde::{Deserialize, Deserializer, Serialize};

use crate::tokenizer::{
    Result, SplitDelimiterBehavior,
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

    /// Pipeline canonicalization. A recognized whole-covering GPT regex shipped
    /// as `(invert=true, behavior=Removed)` — the tiktoken-conversion convention
    /// used by cl100k/o200k — is byte-exactly equivalent to `(invert=false,
    /// Isolated)`, the form the native FSM fast path requires (the inverted match
    /// set is the gaps, and these patterns leave no gaps). Rewrite to it so
    /// cl100k/o200k route to `fsm_cl100k`/`fsm_o200k` instead of the SysRegex fallback.
    pub(crate) fn canonicalized_for_pipeline(self) -> Result<Self> {
        use crate::tokenizer::SplitDelimiterBehavior::{Isolated, Removed};
        if self.fsm.is_some() && self.invert && self.behavior == Removed {
            Split::new(self.pattern, Isolated, false)
        } else {
            Ok(self)
        }
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

    use SplitDelimiterBehavior::*;

    const GPT2_GOLDEN: &[(&str, (u32, u32))] = &[
        ("The", (0, 3)),
        (" quick", (3, 9)),
        (" brown", (9, 15)),
        (" fox", (15, 19)),
        (" 123", (19, 23)),
        ("!!!", (23, 26)),
        (" ", (26, 27)),
        (" double", (27, 34)),
        (" ", (34, 35)),
        (" spaces", (35, 42)),
        ("\t", (42, 43)),
        ("and", (43, 46)),
        (" tabs", (46, 51)),
        (".", (51, 52)),
        (" don", (52, 56)),
        ("'t", (56, 58)),
        (" Naïve", (58, 65)),
        (" café", (65, 71)),
        (".", (71, 72)),
        (" ", (72, 73)),
    ];
    const CL100K_GOLDEN: &[(&str, (u32, u32))] = &[
        ("The", (0, 3)),
        (" quick", (3, 9)),
        (" brown", (9, 15)),
        (" fox", (15, 19)),
        (" ", (19, 20)),
        ("123", (20, 23)),
        ("!!!", (23, 26)),
        (" ", (26, 27)),
        (" double", (27, 34)),
        (" ", (34, 35)),
        (" spaces", (35, 42)),
        ("\tand", (42, 46)),
        (" tabs", (46, 51)),
        (".", (51, 52)),
        (" don", (52, 56)),
        ("'t", (56, 58)),
        (" Naïve", (58, 65)),
        (" café", (65, 71)),
        (".\n\n", (71, 74)),
        ("世界", (74, 80)),
        (" 안녕", (80, 87)),
        (" ", (87, 88)),
    ];
    const O200K_GOLDEN: &[(&str, (u32, u32))] = &[
        ("Mc", (0, 2)),
        ("Donald's", (2, 10)),
        (" i", (10, 12)),
        ("Phone", (12, 17)),
        (" SQLite", (17, 24)),
        (" HELLOWorld", (24, 35)),
        (" camel", (35, 41)),
        ("Case", (41, 45)),
        (" don't", (45, 51)),
        (" I'll", (51, 56)),
        (" We've", (56, 62)),
        (" ", (62, 63)),
        ("3", (63, 64)),
        (".", (64, 65)),
        ("14", (65, 67)),
        (" café", (67, 73)),
        (" Straße", (73, 81)),
        (" 世界", (81, 88)),
        (" 안녕", (88, 95)),
        ("\n\n", (95, 97)),
        (" ", (97, 98)),
        (" Mixed", (98, 104)),
        (" CASE", (104, 109)),
        (" end", (109, 113)),
        (".", (113, 114)),
    ];
    const TEKKEN_GOLDEN: &[(&str, (u32, u32))] = &[
        ("Mc", (0, 2)),
        ("Donald", (2, 8)),
        ("'s", (8, 10)),
        (" i", (10, 12)),
        ("Phone", (12, 17)),
        (" SQLite", (17, 24)),
        (" HELLOWorld", (24, 35)),
        (" camel", (35, 41)),
        ("Case", (41, 45)),
        (" don", (45, 49)),
        ("'t", (49, 51)),
        (" I", (51, 53)),
        ("'ll", (53, 56)),
        (" We", (56, 59)),
        ("'ve", (59, 62)),
        (" ", (62, 63)),
        ("3", (63, 64)),
        (".", (64, 65)),
        ("1", (65, 66)),
        ("4", (66, 67)),
        ("1", (67, 68)),
        ("5", (68, 69)),
        ("9", (69, 70)),
        (" café", (70, 76)),
        (" Straße", (76, 84)),
        (" 世界", (84, 91)),
        (" 안녕", (91, 98)),
        ("\n\n", (98, 100)),
        (" ", (100, 101)),
        (" path", (101, 106)),
        ("/to", (106, 109)),
        ("/file", (109, 114)),
        (" Mixed", (114, 120)),
        (" CASE", (120, 125)),
        (" end", (125, 129)),
        (".", (129, 130)),
    ];
    const QWEN2_GOLDEN: &[(&str, (u32, u32))] = &[
        ("abc", (0, 3)),
        (" ", (3, 4)),
        ("1", (4, 5)),
        ("2", (5, 6)),
        ("3", (6, 7)),
        ("4", (7, 8)),
        ("5", (8, 9)),
        (" don", (9, 13)),
        ("'t", (13, 15)),
        (" A", (15, 17)),
        ("1", (17, 18)),
        ("B", (18, 19)),
        ("2", (19, 20)),
        (" ", (20, 21)),
        ("3", (21, 22)),
        (".", (22, 23)),
        ("1", (23, 24)),
        ("4", (24, 25)),
        (" ", (25, 26)),
        ("1", (26, 27)),
        ("0", (27, 28)),
        ("0", (28, 29)),
        ("%%", (29, 31)),
        (" ", (31, 32)),
        (" double", (32, 39)),
        (" ", (39, 40)),
        (" spaces", (40, 47)),
        ("\ttab", (47, 51)),
        (".\n\n", (51, 54)),
        ("世界", (54, 60)),
        (" ", (60, 61)),
        ("4", (61, 62)),
        ("2", (62, 63)),
    ];

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
    fn splits_by_behavior() {
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
    fn pipeline_gpt2_uses_fsm_and_matches_golden() {
        // The gpt2 pattern is recognized -> the pipeline path routes to the native
        // atomsplit FSM; its output must equal the legacy fancy-regex path.
        let gpt2 = atomsplit::regexes::GPT2;
        let corpus = "The quick brown fox 123!!!  double  spaces\tand tabs. don't Naïve café. ";
        let pretok = Split::new(SplitPattern::Regex(gpt2.into()), Isolated, false).unwrap();
        assert!(pretok.fsm.is_some(), "gpt2 pattern should be recognized");

        assert_eq!(
            pipeline_split(SplitPattern::Regex(gpt2.into()), Isolated, false, corpus),
            GPT2_GOLDEN,
        );
    }

    #[test]
    fn pipeline_cl100k_llama3_uses_fsm_and_matches_golden() {
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

        assert_eq!(
            pipeline_split(SplitPattern::Regex(cl100k.into()), Isolated, false, corpus),
            CL100K_GOLDEN,
        );
    }

    #[test]
    fn pipeline_o200k_uses_fsm_and_matches_golden() {
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

        assert_eq!(
            pipeline_split(SplitPattern::Regex(o200k.into()), Isolated, false, corpus),
            O200K_GOLDEN,
        );
    }

    #[test]
    fn pipeline_tekken_uses_fsm_and_matches_golden() {
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

        assert_eq!(
            pipeline_split(SplitPattern::Regex(tekken.into()), Isolated, false, corpus),
            TEKKEN_GOLDEN,
        );
    }

    #[test]
    fn pipeline_qwen2_uses_fsm_and_matches_golden() {
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

        assert_eq!(
            pipeline_split(SplitPattern::Regex(qwen2.into()), Isolated, false, corpus),
            QWEN2_GOLDEN,
        );
    }

    /// The goal of the literal path: a plain string pattern splits with no regex backend.
    #[test]
    fn a_string_pattern_needs_no_backend() {
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

    // A `Regex` pattern and the `String` pattern it is equivalent to must split alike: on a
    // single-space input `\s+` and " " match the same spans, so the literal path and the regex
    // path have to agree.
    #[cfg(feature = "fancy-regex")] // needs a system-regex backend
    #[test]
    fn regex_string() {
        assert_eq!(
            pipeline_split(
                SplitPattern::Regex(r"\s+".into()),
                SplitDelimiterBehavior::Removed,
                false,
                "Hey, man!"
            ),
            pipeline_split(
                SplitPattern::String(" ".into()),
                SplitDelimiterBehavior::Removed,
                false,
                "Hey, man!"
            ),
        );
    }

    // Splitting on " " and inverted-splitting on "Hello" carve the same input the same way:
    // what one treats as delimiter the other treats as content.
    #[test]
    fn invert() {
        assert_eq!(
            pipeline_split(
                SplitPattern::String(" ".into()),
                SplitDelimiterBehavior::Removed,
                false,
                "Hello Hello Hello"
            ),
            pipeline_split(
                SplitPattern::String("Hello".into()),
                SplitDelimiterBehavior::Removed,
                true,
                "Hello Hello Hello"
            ),
        );
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
