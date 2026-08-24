use crate::pipeline;
use crate::utils::{GptFsm, SysRegex, gpt_fsm};
use atomsplit::literal::Literal;

use crate::tokenizer::{
    Result, SplitDelimiterBehavior,
    pattern::{Invert, Pattern},
};

/// Represents the different patterns that `Split` can use
///
/// Written down externally tagged: `{"String":"..."}` / `{"Regex":"..."}`.
#[derive(Debug, Clone, PartialEq, Eq)]
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

/// Only `pattern`, `behavior` and `invert` are ever written down: `search` and `fsm` are both
/// *derived* from `pattern` by [`Split::new`], so a config carrying them would be a config that can
/// disagree with itself. That is also why the serde in [`super::serialization`] goes through the
/// constructor rather than a struct literal, and why reading one can fail: compiling the pattern
/// can.
#[derive(Debug)]
pub struct Split {
    pub pattern: SplitPattern,
    /// How the pattern is found. A plain string never needs a backend; a regex does, unless it is one
    /// of the GPT patterns the native FSM below covers.
    pub search: Search,
    pub behavior: SplitDelimiterBehavior,
    pub invert: bool,
    /// Native `atomsplit` FSM for a recognized GPT regex (gpt2 / cl100k-Llama-3 / o200k), used on the
    /// pipeline path when `behavior == Isolated && !invert` (how these regexes always ship). Byte-exact
    /// with `regex`; `None` falls back to `regex`.
    fsm: Option<GptFsm>,
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

    /// The native FSM family this pattern was recognised as, if any.
    pub fn gpt_fsm(&self) -> Option<GptFsm> {
        self.fsm
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

    /// `Split::new` compiles every regex through the system backend, so a build without
    /// `fancy-regex` cannot construct deepseek's three patterns -- they are individually
    /// unrecognised by `gpt_fsm` (the pipeline runs them as one native pass). `Split::native`
    /// never asks for an engine.
    #[test]
    fn native_accepts_patterns_new_would_need_an_engine_for() {
        use crate::utils::DEEPSEEK_PATTERNS;

        for pattern in DEEPSEEK_PATTERNS {
            let split = Split::native(
                SplitPattern::Regex(pattern.to_string()),
                SplitDelimiterBehavior::Isolated,
                false,
            )
            .expect("native must not need a regex backend");
            assert!(
                split.gpt_fsm().is_none(),
                "deepseek's patterns are not individually FSM-recognised"
            );
        }

        // The premise: with no backend compiled, `Split::new` rejects exactly these.
        #[cfg(not(feature = "fancy-regex"))]
        for pattern in DEEPSEEK_PATTERNS {
            assert!(
                Split::new(
                    SplitPattern::Regex(pattern.to_string()),
                    SplitDelimiterBehavior::Isolated,
                    false,
                )
                .is_err(),
                "without a backend `Split::new` must fail here -- that is why `native` exists"
            );
        }

        // A recognised pattern still reports its family, so a caller can route it natively.
        let gpt2 = Split::native(
            SplitPattern::Regex(atomsplit::regexes::GPT2.to_string()),
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .unwrap();
        assert_eq!(gpt2.gpt_fsm(), Some(GptFsm::Gpt2));
    }

    #[cfg(feature = "fancy-regex")]
    use SplitDelimiterBehavior::*;

    #[cfg(feature = "fancy-regex")] // only the gated tests below drive it
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
}
