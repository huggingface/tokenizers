use crate::utils::byte_level::{BYTES_CHAR_LOOKUP, CHAR_BYTES_LOOKUP};
use serde::{Deserialize, Serialize};

use crate::tokenizer::{Decoder, PostProcessor, PreTokenizer, Result};
use crate::utils::macro_rules_attribute;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
/// Provides all the necessary steps to handle the BPE tokenization at the byte-level. Takes care
/// of all the required processing steps to transform a UTF-8 string as needed before and after the
/// BPE model does its job.
#[macro_rules_attribute(impl_serde_type!)]
#[non_exhaustive]
pub struct ByteLevel {
    /// Whether to add a leading space to the first word. This allows to treat the leading word
    /// just as any other word.
    pub add_prefix_space: bool,
    /// Whether the post processing step should trim offsets to avoid including whitespaces.
    pub trim_offsets: bool,

    /// Whether to use the standard GPT2 regex for whitespace splitting
    /// Set it to False if you want to use your own splitting.
    #[serde(default = "default_true")]
    pub use_regex: bool,
}

fn default_true() -> bool {
    true
}

impl Default for ByteLevel {
    fn default() -> Self {
        Self {
            add_prefix_space: true,
            trim_offsets: true,
            use_regex: true,
        }
    }
}

impl ByteLevel {
    pub fn new(add_prefix_space: bool, trim_offsets: bool, use_regex: bool) -> Self {
        Self {
            add_prefix_space,
            trim_offsets,
            use_regex,
        }
    }

    pub fn alphabet() -> [char; 256] {
        *BYTES_CHAR_LOOKUP
    }

    #[must_use]
    pub fn add_prefix_space(mut self, v: bool) -> Self {
        self.add_prefix_space = v;
        self
    }

    #[must_use]
    pub fn trim_offsets(mut self, v: bool) -> Self {
        self.trim_offsets = v;
        self
    }

    #[must_use]
    pub fn use_regex(mut self, v: bool) -> Self {
        self.use_regex = v;
        self
    }
}

/// As a `PreTokenizer`, `ByteLevel` is in charge of transforming all the unicode characters into
/// their byte-level counterpart. It also splits the input according to the configured regex.
// TODO: Give the ability to modify this regex
impl PreTokenizer for ByteLevel {}

/// As a `Decoder`, `ByteLevel` is in charge of converting any byte-level characters to their
/// unicode counterpart, before merging everything back into a single String.
/// This decoder will consume the tokens and merge them in one step to alleviate
/// the fact that single token decoded might be a byte not representable as
/// as String.
impl Decoder for ByteLevel {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        let toks = tokens
            .into_iter()
            .flat_map(|t| {
                t.chars()
                    .try_fold(vec![], |mut acc, c| {
                        CHAR_BYTES_LOOKUP.get(&c).map(|b| {
                            acc.push(*b);
                            acc
                        })
                    })
                    .unwrap_or_else(|| t.as_bytes().to_vec())
            })
            .collect::<Vec<u8>>();
        Ok(vec![String::from_utf8_lossy(&toks).to_string()])
    }
}

/// As a [`PostProcessor`], `ByteLevel` adds no token of its own. Its [`Self::trim_offsets`] used
/// to pull each token's offsets in past the leading `Ġ`; there are no offsets to trim now, so the
/// field is read from the config and serialized back out without being acted on.
impl PostProcessor for ByteLevel {
    fn added_tokens(&self, _is_pair: bool) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::tokenizer::Decoder;

    /// Compares against readable literals; `pipeline_splits` hands back owned strings.
    fn assert_splits(got: Vec<(String, (usize, usize))>, want: &[(&str, (usize, usize))]) {
        let got: Vec<(&str, (usize, usize))> = got.iter().map(|(s, o)| (s.as_str(), *o)).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn pre_tokenization() {
        assert_splits(
            pipeline_splits(
                ByteLevel::default().add_prefix_space(false),
                "Hello my friend, how is your day going?",
            ),
            &[
                ("Hello", (0, 5)),
                ("Ġmy", (5, 8)),
                ("Ġfriend", (8, 15)),
                (",", (15, 16)),
                ("Ġhow", (16, 20)),
                ("Ġis", (20, 23)),
                ("Ġyour", (23, 28)),
                ("Ġday", (28, 32)),
                ("Ġgoing", (32, 38)),
                ("?", (38, 39)),
            ],
        );
    }

    #[test]
    fn pre_tokenization_no_regex() {
        assert_splits(
            pipeline_splits(
                ByteLevel::default()
                    .add_prefix_space(false)
                    .use_regex(false),
                "Hello my friend, how is your day going?",
            ),
            &[("HelloĠmyĠfriend,ĠhowĠisĠyourĠdayĠgoing?", (0, 39))],
        );
    }

    #[test]
    fn decoding() {
        let bytelevel = ByteLevel::default().add_prefix_space(false);
        assert_eq!(
            bytelevel
                .decode_chain(
                    vec![
                        "Hello", "Ġmy", "Ġfriend", ",", "Ġhow", "Ġis", "Ġyour", "Ġday", "Ġgoing",
                        "?"
                    ]
                    .into_iter()
                    .map(|s| s.into())
                    .collect::<Vec<String>>()
                )
                .unwrap(),
            vec!["Hello my friend, how is your day going?"]
        );
    }

    #[test]
    fn decode_works_on_separated_tokens() {
        let samples = vec![
            "A Nuskhuri abbreviation of იესუ ქრისტე ( iesu kriste ) \" Jesus Christ \"",
            "An equal number have descenders , like p or q in English \
                 : გ , დ , ე , ვ , კ , ლ , ჟ , ტ , უ , ფ , ღ , ყ , ც",
        ];

        let bytelevel = ByteLevel::default().add_prefix_space(false);
        for sample in samples {
            let separated_tokens = pipeline_splits(bytelevel, sample)
                .iter()
                .flat_map(|(s, _)| s.split("").map(|t| t.into()))
                .collect::<Vec<_>>();
            assert_eq!(
                sample,
                bytelevel.decode_chain(separated_tokens).unwrap().join("")
            );
        }
    }

    #[test]
    fn handling_of_newlines() {
        assert_splits(
            pipeline_splits(
                ByteLevel::default().add_prefix_space(false),
                "Hello there\nHello there",
            ),
            &[
                ("Hello", (0, 5)),
                ("Ġthere", (5, 11)),
                ("Ċ", (11, 12)),
                ("Hello", (12, 17)),
                ("Ġthere", (17, 23)),
            ],
        );
    }

    #[test]
    fn handling_of_multiple_whitespaces() {
        assert_splits(
            pipeline_splits(
                ByteLevel::default().add_prefix_space(false),
                "Hello there       dear",
            ),
            &[
                ("Hello", (0, 5)),
                ("Ġthere", (5, 11)),
                ("ĠĠĠĠĠĠ", (11, 17)),
                ("Ġdear", (17, 22)),
            ],
        );
    }

    #[test]
    fn offsets_when_char_split_up() {
        let input = "i⭢j";
        let splits = pipeline_splits(ByteLevel::default().add_prefix_space(false), input);
        // the projected piece is 6 bytes wide, its span stays the 3 raw bytes of '⭢'
        assert_splits(
            splits.clone(),
            &[("i", (0, 1)), ("âŃ¢", (1, 4)), ("j", (4, 5))],
        );
        // spans still index the original text
        assert_eq!(
            splits
                .iter()
                .map(|(_, o)| &input[o.0..o.1])
                .collect::<Vec<_>>(),
            vec!["i", "⭢", "j"]
        );
    }

    #[test]
    fn decode_unknown_characters() {
        let byte_level = ByteLevel::default();
        assert_eq!(
            byte_level
                .decode_chain(vec![
                    "Hello".into(),
                    "Ġthere".into(),
                    "Ġdear".into(),
                    "Ġfriend!".into(),
                    "Ġ".into(),
                    "[PA D]".into()
                ])
                .unwrap(),
            vec!["Hello there dear friend! [PA D]"]
        );
    }

    /// Splits from the pipeline conversion of `byte_level`, with the raw text of each
    /// range transformed to the byte-level alphabet so it's comparable with the legacy
    /// oracle's output strings.
    fn pipeline_splits(byte_level: ByteLevel, text: &str) -> Vec<(String, (usize, usize))> {
        use std::convert::TryFrom;
        let converted = crate::pipeline::PipelinePreTokenizer::try_from(
            crate::PreTokenizerWrapper::ByteLevel(byte_level),
        )
        .unwrap();
        let mut scratch = crate::pipeline::PreTokenizerScratch::default();
        let mut out = Vec::new();
        crate::pipeline::PreTokenizer::pre_tokenize(&converted, text, &mut scratch, &mut out)
            .unwrap();
        out.iter()
            .map(|s| {
                let transformed = text[s.range()]
                    .bytes()
                    .map(|b| BYTES_CHAR_LOOKUP[b as usize])
                    .collect();
                (transformed, (s.start as usize, s.end as usize))
            })
            .collect()
    }

    const CONVERSION_GOLDEN: &[&[(&str, (usize, usize))]] = &[
        &[
            ("Hello", (0, 5)),
            ("Ġmy", (5, 8)),
            ("Ġfriend", (8, 15)),
            (",", (15, 16)),
            ("Ġhow", (16, 20)),
            ("Ġis", (20, 23)),
            ("Ġyour", (23, 28)),
            ("Ġday", (28, 32)),
            ("Ġgoing", (32, 38)),
            ("?", (38, 39)),
        ],
        &[
            ("Hello", (0, 5)),
            ("Ġthere", (5, 11)),
            ("Ċ", (11, 12)),
            ("Hello", (12, 17)),
            ("Ġthere", (17, 23)),
        ],
        &[
            ("Hello", (0, 5)),
            ("Ġthere", (5, 11)),
            ("ĠĠĠĠĠĠ", (11, 17)),
            ("Ġdear", (17, 22)),
        ],
        &[("Ġleading", (0, 8)), ("Ġspace", (8, 14))],
        &[("trailing", (0, 8)), ("Ġspace", (8, 14)), ("ĠĠĠ", (14, 17))],
        &[("i", (0, 1)), ("âŃ¢", (1, 4)), ("j", (4, 5))],
        &[
            ("ä¸Ńæĸĩ", (0, 6)),
            ("Ġtext", (6, 11)),
            ("Ġ123", (11, 15)),
            (",", (15, 16)),
            ("Ġmixed", (16, 22)),
            ("!", (22, 23)),
            ("ĠðŁ¤Ĺ", (23, 28)),
            ("Ġemoji", (28, 34)),
        ],
        &[
            ("I", (0, 1)),
            ("'m", (1, 3)),
            ("Ġcan", (3, 7)),
            ("'t", (7, 9)),
            ("Ġwe", (9, 12)),
            ("'ve", (12, 15)),
            ("Ġthey", (15, 20)),
            ("'ll", (20, 23)),
            ("Ġit", (23, 26)),
            ("'s", (26, 28)),
        ],
        &[
            ("tabs", (0, 4)),
            ("ĉ", (4, 5)),
            ("and", (5, 8)),
            ("č", (8, 9)),
            ("Ċ", (9, 10)),
            ("newlines", (10, 18)),
        ],
        &[
            ("cafÃ©", (0, 5)),
            ("ĠÃ¼ber", (5, 11)),
            ("ĠnaÃ¯ve", (11, 18)),
        ],
        &[("!!!???...", (0, 9))],
        &[("single", (0, 6))],
    ];

    const CONVERSION_TEXTS: &[&str] = &[
        "Hello my friend, how is your day going?",
        "Hello there\nHello there",
        "Hello there       dear",
        " leading space",
        "trailing space   ",
        "i⭢j",
        "中文 text 123, mixed! 🤗 emoji",
        "I'm can't we've they'll it's",
        "tabs\tand\r\nnewlines",
        "café über naïve",
        "!!!???...",
        "single",
    ];

    #[test]
    fn pipeline_conversion_matches_golden_splits() {
        let byte_level = ByteLevel::default().add_prefix_space(false);
        assert_eq!(
            CONVERSION_GOLDEN.len(),
            CONVERSION_TEXTS.len(),
            "one expectation per text"
        );
        for (text, want) in CONVERSION_TEXTS.iter().zip(CONVERSION_GOLDEN) {
            let splits = pipeline_splits(byte_level, text);
            let got: Vec<(&str, (usize, usize))> =
                splits.iter().map(|(s, o)| (s.as_str(), *o)).collect();
            assert_eq!(got.as_slice(), *want, "diverged on {text:?}");
        }
    }

    #[test]
    fn pipeline_conversion_no_regex_is_identity_split() {
        let byte_level = ByteLevel::default()
            .add_prefix_space(false)
            .use_regex(false);
        let text = "Hello my friend, how is your day going?";
        assert_splits(
            pipeline_splits(byte_level, text),
            &[("HelloĠmyĠfriend,ĠhowĠisĠyourĠdayĠgoing?", (0, 39))],
        );
    }

    #[test]
    fn pipeline_conversion_rejects_add_prefix_space() {
        // The range-based pipeline can't prepend text; converting must fail loudly
        // rather than silently produce different splits than the legacy path.
        use std::convert::TryFrom;
        let byte_level = ByteLevel::default().add_prefix_space(true);
        assert!(
            crate::pipeline::PipelinePreTokenizer::try_from(crate::PreTokenizerWrapper::ByteLevel(
                byte_level
            ))
            .is_err()
        );
    }

    #[test]
    fn deserialization() {
        // Before use_regex
        let byte_level: ByteLevel = serde_json::from_str(
            r#"{"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false}"#,
        )
        .unwrap();
        assert!(byte_level.use_regex);

        // Loading works, new future BC test.
        let byte_level: ByteLevel = serde_json::from_str(
            r#"{"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": true}"#,
        )
        .unwrap();
        assert!(byte_level.use_regex);

        let byte_level: ByteLevel = serde_json::from_str(
            r#"{"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": false}"#,
        )
        .unwrap();
        assert!(!byte_level.use_regex);
    }
}
