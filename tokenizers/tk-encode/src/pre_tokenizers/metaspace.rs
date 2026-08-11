use crate::normalizers::metaspace::MetaspaceNormalizer;
use crate::pre_tokenizers::PreTokenizerWrapper;
use crate::pre_tokenizers::split::Split;
use crate::tokenizer::{Decoder, PreTokenizer, Result, SplitDelimiterBehavior};
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

impl PreTokenizer for Metaspace {}

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

/// Rebuilds a `Metaspace` pre-tokenizer as the two pipeline steps that together do its job:
///   - a [`MetaspaceNormalizer`], writing the `▁` delimiters into the text,
///   - a [`Split`] on that delimiter, cutting the text into words.
///
/// [`crate::normalizers::metaspace`] explains what those delimiters are for, and why the pipeline
/// wants the two steps apart.
///
/// `None` when the pre-tokenizer is not a `Metaspace`, or a `Metaspace` with settings we can reproduce with a Normalizer + Split
/// The caller then converts the pre-tokenizer the usual way, and rejects residual `Metaspace`. An unsupported config leaves
/// the model out of the pipeline instead of quietly encoding it differently.
pub(crate) fn to_normalizer_and_split(
    pre_tokenizer: Option<&PreTokenizerWrapper>,
) -> Option<(MetaspaceNormalizer, Split)> {
    match pre_tokenizer {
        Some(PreTokenizerWrapper::Metaspace(metaspace)) => normalizer_and_split(metaspace, false),
        // The same, with the whitespace thrown away first. t5 and albert ship this shape.
        Some(PreTokenizerWrapper::Sequence(sequence)) => match sequence.as_ref() {
            [
                PreTokenizerWrapper::WhitespaceSplit(_),
                PreTokenizerWrapper::Metaspace(metaspace),
            ] => normalizer_and_split(metaspace, true),
            _ => None,
        },
        _ => None,
    }
}

fn normalizer_and_split(
    metaspace: &Metaspace,
    drop_whitespace: bool,
) -> Option<(MetaspaceNormalizer, Split)> {
    // `split: false` writes the delimiters but never cuts the text, so there is no `Split` step to
    // hand back.
    if !metaspace.split {
        return None;
    }
    let prepend = match metaspace.prepend_scheme {
        PrependScheme::Always => true,
        PrependScheme::Never => false,
        // `First` writes the delimiter only on the piece at the very start of the text it came from.
        // A normalizer is handed one chunk at a time, without that context.
        PrependScheme::First => return None,
    };
    // Removes whitespaces and does not prepend words: nothing would show where words begin
    // The output is one big continuous blob of words hlued together
    if drop_whitespace && !prepend {
        return None;
    }
    let delimiter = metaspace.replacement;
    Some((
        MetaspaceNormalizer::new(delimiter, prepend, drop_whitespace),
        // `MergedWithNext` keeps each delimiter attached to the word it opens (`▁hello`), which is
        // how SentencePiece vocabularies spell their tokens. A one-character literal always
        // compiles, so `ok()?` is not hiding a case worth reporting.
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

    use super::*;

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

        fn pre_tokenizer_from(json: &str) -> PreTokenizerWrapper {
            serde_json::from_str(json).unwrap()
        }

        /// t5 and albert: throw the whitespace away, then start every word with `▁`.
        const T5: &str = r#"{"type":"Sequence","pretokenizers":[{"type":"WhitespaceSplit"},{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}]}"#;
        /// A `Metaspace` on its own: each space becomes `▁`, tabs and newlines stay.
        const BARE: &str =
            r#"{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}"#;
        /// Nothing ties the delimiter to `▁`, and a 1-byte one takes a different code path in
        /// `Split`, so keep an ASCII delimiter under test too.
        const ASCII_DELIMITER: &str =
            r#"{"type":"Metaspace","replacement":"_","prepend_scheme":"always","split":true}"#;

        /// Every kind of gap, plus text that already holds the delimiter.
        const TEXTS: &[&str] = &[
            "hello world",
            "hello   world",
            " leading",
            "trailing ",
            "  both  ",
            "one\ttab\nand a newline",
            "\tleading tab",
            // A gap that is whitespace to `char::is_whitespace` but not to an ASCII scan:
            // a no-break space and an ideographic space.
            "nbsp\u{a0}gap and\u{3000}an ideographic space",
            "▁already marked",
            "a▁b c",
            "▁▁▁a b",
            "▁",
            "_underscored_ text",
            "single",
            "   ",
            "",
        ];

        /// The words `normalize` + `split` carve out of each of [`TEXTS`], frozen from the
        /// `Metaspace` pre-tokenizer this shape was built to reproduce.
        fn assert_words_match(json: &str, expected: &[&[&str]]) {
            let declared = pre_tokenizer_from(json);
            let (normalizer, split) =
                to_normalizer_and_split(Some(&declared)).expect("this shape is supported");
            assert_eq!(expected.len(), TEXTS.len(), "one expectation per text");
            for (text, expected) in TEXTS.iter().zip(expected) {
                let normalized = pipeline::Normalizer::normalize(&normalizer, text).unwrap();
                let mut scratch = pipeline::PreTokenizerScratch::default();
                let mut spans = Vec::new();
                pipeline::PreTokenizer::pre_tokenize(&split, &normalized, &mut scratch, &mut spans)
                    .unwrap();
                let words: Vec<&str> = spans.iter().map(|s| &normalized[s.range()]).collect();
                assert_eq!(words, *expected, "{text:?}");
            }
        }

        const T5_WORDS: &[&[&str]] = &[
            &["▁hello", "▁world"],
            &["▁hello", "▁world"],
            &["▁leading"],
            &["▁trailing"],
            &["▁both"],
            &["▁one", "▁tab", "▁and", "▁a", "▁newline"],
            &["▁leading", "▁tab"],
            &["▁nbsp", "▁gap", "▁and", "▁an", "▁ideographic", "▁space"],
            &["▁already", "▁marked"],
            &["▁a", "▁b", "▁c"],
            &["▁", "▁", "▁a", "▁b"],
            &["▁"],
            &["▁_underscored_", "▁text"],
            &["▁single"],
            &[],
            &[],
        ];
        const BARE_WORDS: &[&[&str]] = &[
            &["▁hello", "▁world"],
            &["▁hello", "▁", "▁", "▁world"],
            &["▁leading"],
            &["▁trailing", "▁"],
            &["▁", "▁both", "▁", "▁"],
            &["▁one\ttab\nand", "▁a", "▁newline"],
            &["▁\tleading", "▁tab"],
            &["▁nbsp\u{a0}gap", "▁and\u{3000}an", "▁ideographic", "▁space"],
            &["▁already", "▁marked"],
            &["▁a", "▁b", "▁c"],
            &["▁", "▁", "▁a", "▁b"],
            &["▁"],
            &["▁_underscored_", "▁text"],
            &["▁single"],
            &["▁", "▁", "▁"],
            &[],
        ];
        const ASCII_DELIMITER_WORDS: &[&[&str]] = &[
            &["_hello", "_world"],
            &["_hello", "_", "_", "_world"],
            &["_leading"],
            &["_trailing", "_"],
            &["_", "_both", "_", "_"],
            &["_one\ttab\nand", "_a", "_newline"],
            &["_\tleading", "_tab"],
            &["_nbsp\u{a0}gap", "_and\u{3000}an", "_ideographic", "_space"],
            &["_▁already", "_marked"],
            &["_a▁b", "_c"],
            &["_▁▁▁a", "_b"],
            &["_▁"],
            &["_underscored", "_", "_text"],
            &["_single"],
            &["_", "_", "_"],
            &[],
        ];

        #[test]
        fn t5_shape_matches_its_pre_tokenizer() {
            assert_words_match(T5, T5_WORDS);
        }

        #[test]
        fn bare_metaspace_matches_its_pre_tokenizer() {
            assert_words_match(BARE, BARE_WORDS);
        }

        #[test]
        fn ascii_delimiter_matches_its_pre_tokenizer() {
            assert_words_match(ASCII_DELIMITER, ASCII_DELIMITER_WORDS);
        }

        #[test]
        fn refuses_what_it_cannot_reproduce() {
            let refused = [
                // Delimiters written but the text never cut: no `Split` step to hand back.
                (
                    "a metaspace that does not split",
                    r#"{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":false}"#,
                ),
                // `first` marks a piece only when it opened the text it came from; a normalizer is
                // given chunks, not their position.
                (
                    "a metaspace that prepends to the first word only",
                    r#"{"type":"Metaspace","replacement":"▁","prepend_scheme":"first","split":true}"#,
                ),
                // Whitespace gone and no delimiter written: nothing left to show where words start.
                (
                    "dropped whitespace and no prepending",
                    r#"{"type":"Sequence","pretokenizers":[{"type":"WhitespaceSplit"},{"type":"Metaspace","replacement":"▁","prepend_scheme":"never","split":true}]}"#,
                ),
                // Not a metaspace shape at all.
                ("a bare whitespace split", r#"{"type":"WhitespaceSplit"}"#),
            ];
            for (name, json) in refused {
                assert!(
                    to_normalizer_and_split(Some(&pre_tokenizer_from(json))).is_none(),
                    "{name}"
                );
            }
            assert!(to_normalizer_and_split(None).is_none(), "no pre-tokenizer");
        }

        /// The real files, so a config shape drifting out of the two above shows up here instead of
        /// silently skipping the pipeline oracle for these models. Skipped when they are not fetched.
        #[test]
        fn real_configs_convert() {
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
