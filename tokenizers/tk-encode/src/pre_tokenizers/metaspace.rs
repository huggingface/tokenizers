use crate::normalizers::metaspace::MetaspaceNormalizer;
use crate::pre_tokenizers::PreTokenizerWrapper;
use crate::pre_tokenizers::split::Split;
use crate::tokenizer::{Decoder, PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior};
use serde::{Deserialize, Deserializer, Serialize, de};

/// Enum representing options for the metaspace prepending scheme.
#[derive(Debug, Clone, PartialEq, Serialize, Eq, Deserialize, Copy)]
#[serde(rename_all = "snake_case")]
pub enum PrependScheme {
    /// Specifies that the scheme should be prepended only once, on the first split.
    First,
    /// Specifies that the space should not be prepended.
    Never,
    /// Specifies that the scheme should always be prepended.
    Always,
}

impl std::fmt::Display for PrependScheme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.serialize(f)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Eq)]
/// Replaces all the whitespaces by the provided meta character and then
/// splits on this character
#[serde(tag = "type")]
pub struct Metaspace {
    replacement: char,
    pub prepend_scheme: PrependScheme,
    pub split: bool,
    #[serde(skip)]
    str_rep: String,
}

impl<'de> Deserialize<'de> for Metaspace {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        enum Type {
            Metaspace,
        }

        fn default_prepend_scheme_value() -> PrependScheme {
            PrependScheme::Always
        }

        #[derive(Deserialize)]
        pub struct MetaspaceHelper {
            #[serde(rename = "type")]
            _type: Type,
            replacement: char,

            pub add_prefix_space: Option<bool>,
            #[serde(default = "default_prepend_scheme_value")]
            pub prepend_scheme: PrependScheme,
            pub split: Option<bool>,
            #[serde(rename = "str_rep")]
            _str_rep: Option<String>,
        }

        let mut helper = MetaspaceHelper::deserialize(deserializer)?;
        if let Some(false) = helper.add_prefix_space {
            if helper.prepend_scheme != PrependScheme::Never {
                return Err(de::Error::custom(
                    "add_prefix_space does not match declared prepend_scheme",
                ));
            }
            helper.prepend_scheme = PrependScheme::Never;
        }
        let instance = Self::new(
            helper.replacement,
            helper.prepend_scheme,
            helper.split.unwrap_or(true),
        );
        Ok(instance)
    }
}

impl Metaspace {
    pub fn new(replacement: char, prepend_scheme: PrependScheme, split: bool) -> Self {
        Self {
            replacement,
            str_rep: replacement.to_string(),
            prepend_scheme,
            split,
        }
    }

    pub fn get_replacement(&self) -> char {
        self.replacement
    }

    pub fn set_replacement(&mut self, replacement: char) {
        self.replacement = replacement;
        self.str_rep = replacement.to_string();
    }

    pub fn get_split(&self) -> bool {
        self.split
    }

    pub fn set_split(&mut self, split: bool) {
        self.split = split;
    }

    pub fn get_prepend_scheme(&self) -> PrependScheme {
        self.prepend_scheme
    }

    pub fn set_prepend_scheme(&mut self, scheme: PrependScheme) {
        self.prepend_scheme = scheme;
    }
}

impl Default for Metaspace {
    fn default() -> Self {
        Self::new('▁', PrependScheme::Always, true)
    }
}

impl PreTokenizer for Metaspace {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, mut normalized| {
            normalized.replace(' ', &self.str_rep)?;
            match self.prepend_scheme {
                PrependScheme::Always => {
                    if !normalized.get().starts_with(self.replacement) {
                        normalized.prepend(&self.str_rep);
                    }
                }
                PrependScheme::First => {
                    if !normalized.get().starts_with(self.replacement)
                        && normalized.offsets_original().0 == 0
                    {
                        normalized.prepend(&self.str_rep);
                    }
                }
                PrependScheme::Never => {}
            };
            if self.split {
                normalized.split(self.replacement, SplitDelimiterBehavior::MergedWithNext)
            } else {
                Ok(vec![normalized])
            }
        })
    }
}

impl Decoder for Metaspace {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        Ok(tokens
            .iter()
            .enumerate()
            .map(|(i, token)| {
                token
                    .chars()
                    .flat_map(|c| {
                        if c == self.replacement {
                            if i == 0 && self.prepend_scheme != PrependScheme::Never {
                                None
                            } else {
                                Some(' ')
                            }
                        } else {
                            Some(c)
                        }
                    })
                    .collect::<String>()
            })
            .collect())
    }
}

/// Splits [`PreTokenizerWrapper::Metaspace`] into two steps:
///   - [`MetaspaceNormalizer`] that rewrites spaces and whitespaces
///   - [`Split`] pretokenizers that splits words based on the metaspace delimiter
///
/// Returns `None` if the pre tokenizer is not a metaspace
pub(crate) fn to_normalizer_and_split(
    pre_tokenizer: Option<&PreTokenizerWrapper>,
) -> Option<(MetaspaceNormalizer, Split)> {
    match pre_tokenizer {
        Some(PreTokenizerWrapper::Metaspace(metaspace)) => from_metaspace(metaspace, false),
        // The same, with the whitespace thrown away first. t5 and albert ship this shape.
        Some(PreTokenizerWrapper::Sequence(sequence)) => match sequence.as_ref() {
            [
                PreTokenizerWrapper::WhitespaceSplit(_),
                PreTokenizerWrapper::Metaspace(metaspace),
            ] => from_metaspace(metaspace, true),
            _ => None,
        },
        _ => None,
    }
}

fn from_metaspace(
    metaspace: &Metaspace,
    drop_whitespace: bool,
) -> Option<(MetaspaceNormalizer, Split)> {
    if !metaspace.split {
        return None;
    }
    let prepend = match metaspace.prepend_scheme {
        PrependScheme::Always => true,
        PrependScheme::Never => false,
        PrependScheme::First => return None,
    };
    if drop_whitespace && !prepend {
        return None;
    }
    let delimiter = metaspace.replacement;
    Some((
        MetaspaceNormalizer::new(delimiter, prepend, drop_whitespace),
        Split::new(
            delimiter.to_string(),
            SplitDelimiterBehavior::MergedWithNext,
            false,
        )
        .ok()?,
    ))
}

#[cfg(test)]
mod tests {
    use regex::Regex;

    use super::*;
    use crate::{OffsetReferential, OffsetType};

    #[test]
    fn serialization() {
        let metaspace = Metaspace::new('_', PrependScheme::Always, true);
        let metaspace_s =
            r#"{"type":"Metaspace","replacement":"_","prepend_scheme":"always","split":true}"#;
        assert_eq!(serde_json::to_string(&metaspace).unwrap(), metaspace_s);
        assert_eq!(
            serde_json::from_str::<Metaspace>(metaspace_s).unwrap(),
            metaspace
        );

        // Also check it can deserialize previous versions
        let metaspace_s = r#"{"type":"Metaspace","replacement":"_","add_prefix_space":false,"prepend_scheme":"always"}"#;
        assert!(serde_json::from_str::<Metaspace>(metaspace_s).is_err(),);

        let metaspace = Metaspace::new('_', PrependScheme::Always, true);
        let metaspace_s = r#"{"type":"Metaspace","str_rep":"_","replacement":"_","add_prefix_space":true,"prepend_scheme":"always"}"#;
        assert_eq!(
            serde_json::from_str::<Metaspace>(metaspace_s).unwrap(),
            metaspace
        );

        let metaspace_parsed: Metaspace = serde_json::from_str(
            r#"{"type":"Metaspace","replacement":"_","add_prefix_space":true}"#,
        )
        .unwrap();
        assert_eq!(metaspace_parsed, metaspace);
    }

    #[test]
    fn basic() {
        let pretok = Metaspace::new('▁', PrependScheme::Always, true);
        let mut pretokenized = PreTokenizedString::from("Hey friend!");
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Normalized, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![("▁Hey", (0, 6)), ("▁friend!", (6, 16))]
        );
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![("▁Hey", (0, 3)), ("▁friend!", (3, 11))]
        );
    }

    #[test]
    fn multiple_spaces() {
        let pretok = Metaspace::new('▁', PrependScheme::Always, true);
        let mut pretokenized = PreTokenizedString::from("Hey   friend!");
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Normalized, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![
                ("▁Hey", (0, 6)),
                ("▁", (6, 9)),
                ("▁", (9, 12)),
                ("▁friend!", (12, 22)),
            ]
        );
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![
                ("▁Hey", (0, 3)),
                ("▁", (3, 4)),
                ("▁", (4, 5)),
                ("▁friend!", (5, 13)),
            ]
        );
    }

    #[test]
    fn non_legacy_meta_space() {
        let mut pretok = Metaspace::new('▁', PrependScheme::Always, true);
        pretok.set_prepend_scheme(PrependScheme::Always);
        assert_eq!(pretok, Metaspace::new('▁', PrependScheme::Always, true));

        pretok.set_prepend_scheme(PrependScheme::Never);
        assert_eq!(pretok, Metaspace::new('▁', PrependScheme::Never, true));

        pretok.set_prepend_scheme(PrependScheme::First);
        assert_eq!(pretok, Metaspace::new('▁', PrependScheme::First, true));

        let pretok = Metaspace::new('▁', PrependScheme::First, false);
        let mut pretokenized = PreTokenizedString::from("Hey my friend <s>how▁are you");
        let re_ref = Regex::new(r"(<s>)").unwrap();
        pretokenized
            .split(|_, sequence| sequence.split(&re_ref, SplitDelimiterBehavior::Isolated))
            .expect("Bad split");

        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Normalized, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![
                ("▁Hey▁my▁friend▁", (0, 23)),
                ("<s>", (23, 26)),
                ("how▁are▁you", (26, 41))
            ]
        );
        let pretok = Metaspace::new('▁', PrependScheme::Always, true);
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Normalized, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![
                ("▁Hey", (0, 6)),
                ("▁my", (6, 11)),
                ("▁friend", (11, 20)),
                ("▁", (20, 23)),
                ("▁<s>", (23, 29)),
                ("▁how", (29, 35)),
                ("▁are", (35, 41)),
                ("▁you", (41, 47))
            ]
        );

        let pretok = Metaspace::new('▁', PrependScheme::First, false);
        let mut pretokenized = PreTokenizedString::from(" Hey <s>how"); // test with prefix
        pretokenized
            .split(|_, sequence| sequence.split(&re_ref, SplitDelimiterBehavior::Isolated))
            .expect("Bad split");
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Normalized, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![("▁Hey▁", (0, 9)), ("<s>", (9, 12)), ("how", (12, 15))]
        );

        let mut pretokenized = PreTokenizedString::from(" Hey <s>how <s>are <s> you"); // test with many splits
        pretokenized
            .split(|_, sequence| sequence.split(&re_ref, SplitDelimiterBehavior::Isolated))
            .expect("Bad split");
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Normalized, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![
                ("▁Hey▁", (0, 9)),
                ("<s>", (9, 12)),
                ("how▁", (12, 18)),
                ("<s>", (18, 21)),
                ("are▁", (21, 27)),
                ("<s>", (27, 30)),
                ("▁you", (30, 36))
            ]
        );
    }
    #[test]
    fn decode() {
        let decoder = Metaspace::new('▁', PrependScheme::Always, true);
        let res = decoder
            .decode_chain(vec!["▁Hey".into(), "▁friend!".into()])
            .unwrap();
        assert_eq!(res, vec!["Hey", " friend!"]);

        let decoder = Metaspace::new('▁', PrependScheme::Never, true);
        let res = decoder
            .decode_chain(vec!["▁Hey".into(), "▁friend!".into()])
            .unwrap();
        assert_eq!(res, vec![" Hey", " friend!"]);
    }

    mod to_normalizer_and_split {
        use super::*;
        use crate::tokenizer::pipeline;

        fn declared(json: &str) -> PreTokenizerWrapper {
            serde_json::from_str(json).unwrap()
        }

        /// t5 and albert: throw the whitespace away, then start every word with `▁`.
        const T5: &str = r#"{"type":"Sequence","pretokenizers":[{"type":"WhitespaceSplit"},{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}]}"#;
        /// A `Metaspace` on its own: each space becomes `▁`, tabs and newlines stay.
        const BARE: &str =
            r#"{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}"#;

        /// Every kind of gap, plus text that already holds the delimiter.
        const TEXTS: &[&str] = &[
            "hello world",
            "hello   world",
            " leading",
            "trailing ",
            "  both  ",
            "one\ttab\nand a newline",
            "▁already marked",
            "a▁b c",
            "▁▁▁a b",
            "single",
            "   ",
            "",
        ];

        /// Normalizing and then splitting must produce exactly the words the declared pre-tokenizer
        /// produces on its own — it is the tokenizer's own output, so it is the answer.
        fn assert_words_match_the_pre_tokenizer(json: &str) {
            let declared = declared(json);
            let (normalizer, split) =
                to_normalizer_and_split(Some(&declared)).expect("this shape is supported");
            for text in TEXTS {
                let mut legacy = PreTokenizedString::from(*text);
                declared.pre_tokenize(&mut legacy).unwrap();
                let expected: Vec<&str> = legacy
                    .get_splits(OffsetReferential::Normalized, OffsetType::Byte)
                    .iter()
                    .map(|(word, _, _)| *word)
                    .collect();

                let normalized = pipeline::Normalizer::normalize(&normalizer, text).unwrap();
                let mut spans = Vec::new();
                pipeline::PreTokenizer::pre_tokenize(&split, &normalized, &mut spans).unwrap();
                let words: Vec<&str> = spans.iter().map(|s| &normalized[s.range()]).collect();
                assert_eq!(words, expected, "{text:?}");
            }
        }

        #[test]
        fn the_t5_shape_matches_its_pre_tokenizer() {
            assert_words_match_the_pre_tokenizer(T5);
        }

        #[test]
        fn a_bare_metaspace_matches_its_pre_tokenizer() {
            assert_words_match_the_pre_tokenizer(BARE);
        }

        #[test]
        fn refuses_what_it_cannot_reproduce() {
            let refused = [
                // A metaspace that keeps whole sentences is a different pipeline.
                (
                    "a metaspace that does not split",
                    r#"{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":false}"#,
                ),
                // `first` picks the word by where it sat in the untouched input, which is lost here.
                (
                    "a metaspace that prepends to the first word only",
                    r#"{"type":"Metaspace","replacement":"▁","prepend_scheme":"first","split":true}"#,
                ),
                // With the whitespace gone and no delimiter written, nothing marks where a word starts.
                (
                    "dropped whitespace and no prepending",
                    r#"{"type":"Sequence","pretokenizers":[{"type":"WhitespaceSplit"},{"type":"Metaspace","replacement":"▁","prepend_scheme":"never","split":true}]}"#,
                ),
                // Not a metaspace shape at all.
                ("a bare whitespace split", r#"{"type":"WhitespaceSplit"}"#),
            ];
            for (name, json) in refused {
                assert!(
                    to_normalizer_and_split(Some(&declared(json))).is_none(),
                    "{name}"
                );
            }
            assert!(to_normalizer_and_split(None).is_none(), "no pre-tokenizer");
        }

        /// The real files, so a config shape drifting out of the two above shows up here instead of
        /// silently skipping the pipeline oracle for these models. Skipped when they are not fetched.
        #[test]
        fn the_real_configs_convert() {
            for file in ["t5-base.json", "albert-base-v1-tokenizer.json"] {
                let path = format!("../data/{file}");
                if !std::path::Path::new(&path).exists() {
                    continue;
                }
                let tokenizer = crate::Tokenizer::from_file(&path).unwrap();
                assert!(
                    to_normalizer_and_split(tokenizer.get_pre_tokenizer()).is_some(),
                    "{file} should convert"
                );
            }
        }
    }
}
