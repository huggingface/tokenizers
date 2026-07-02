use crate::pipeline;
use crate::utils::{MultiRegex, SysRegex};
use serde::{Deserialize, Deserializer, Serialize};

use crate::tokenizer::{
    pattern::{Invert, Pattern},
    PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior,
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

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub struct Split {
    pub pattern: SplitPattern,
    #[serde(skip)]
    pub regex: SysRegex,
    pub behavior: SplitDelimiterBehavior,
    pub invert: bool,
    /// Fast pure-DFA matcher for recognized GPT patterns (pipeline path only);
    /// `None` falls back to `regex`. Span-equivalent to `regex` by construction.
    #[serde(skip)]
    multi: Option<MultiRegex>,
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
        let (regex, multi) = match &pattern {
            SplitPattern::String(s) => (SysRegex::new(&regex::escape(s))?, None),
            SplitPattern::Regex(r) => (
                SysRegex::new(r)?,
                MultiRegex::for_gpt_pattern(r).transpose()?,
            ),
        };

        Ok(Self {
            pattern,
            regex,
            behavior,
            invert,
            multi,
        })
    }
}

impl PreTokenizer for Split {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        if self.invert {
            pretokenized.split(|_, normalized| normalized.split(Invert(&self.regex), self.behavior))
        } else {
            pretokenized.split(|_, normalized| normalized.split(&self.regex, self.behavior))
        }
    }
}

impl pipeline::PreTokenizer for Split {
    fn pre_tokenize(&self, text: &str, out: &mut Vec<pipeline::Split>) -> Result<()> {
        let matches: Vec<((usize, usize), bool)> = match &self.multi {
            Some(multi) => {
                let mut segments = Vec::new();
                let mut prev = 0;
                for (start, end) in multi.split_ranges(text) {
                    if prev < start {
                        segments.push(((prev, start), self.invert)); // gap
                    }
                    segments.push(((start, end), !self.invert)); // match
                    prev = end;
                }
                if prev < text.len() {
                    segments.push(((prev, text.len()), self.invert));
                }
                segments
            }
            None if self.invert => Invert(&self.regex).find_matches(text)?,
            None => (&self.regex).find_matches(text)?,
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
        let mut splits = Vec::new();
        crate::pipeline::PreTokenizer::pre_tokenize(&pretok, text, &mut splits).unwrap();
        splits
            .iter()
            .map(|s| (&text[s.range()], (s.start, s.end)))
            .collect()
    }

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
    fn pipeline_gpt2_uses_multiregex_and_matches_legacy() {
        // The gpt2 pattern is recognized -> the pipeline path runs the fast
        // MultiRegex; its output must equal the legacy fancy-regex path.
        let gpt2 = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
        let corpus = "The quick brown fox 123!!!  double  spaces\tand tabs. don't Naïve café. ";
        let pretok = Split::new(SplitPattern::Regex(gpt2.into()), Isolated, false).unwrap();
        assert!(pretok.multi.is_some(), "gpt2 pattern should be recognized");

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
