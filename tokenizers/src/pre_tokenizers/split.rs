use crate::utils::SysRegex;
use serde::{Deserialize, Deserializer, Serialize};

use crate::tokenizer::{
    pattern::Invert, PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior,
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
    pattern: SplitPattern,
    #[serde(skip)]
    regex: SysRegex,
    behavior: SplitDelimiterBehavior,
    invert: bool,
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
        let regex = match &pattern {
            SplitPattern::String(s) => SysRegex::new(&regex::escape(s))?,
            SplitPattern::Regex(r) => SysRegex::new(r)?,
        };

        Ok(Self {
            pattern,
            regex,
            behavior,
            invert,
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
