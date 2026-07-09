use crate::pipeline;
use crate::tokenizer::{Decoder, PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior};
use serde::{de, Deserialize, Deserializer, Serialize};

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

impl Metaspace {
    /// Pipeline-side form of [`PreTokenizer::pre_tokenize`], for one pre-token
    /// `piece`. Metaspace edits text (space → replacement, prepend), which the
    /// range-based [`pipeline::PreTokenizer`] cannot express, so the rewritten
    /// text is appended to the caller's `scratch` and the emitted
    /// [`pipeline::Split`]s are ranges into `scratch`.
    ///
    /// A single pass replaces, prepends, and splits at once: every replacement
    /// char landing in `scratch` starts a new split, which is exactly
    /// `SplitDelimiterBehavior::MergedWithNext` on the rewritten text.
    ///
    /// `PrependScheme::First` is unsupported (rejected at pipeline build): it
    /// depends on the piece's offset in the original input, which the pipeline
    /// does not track.
    pub(crate) fn rewrite(
        &self,
        piece: &str,
        scratch: &mut String,
        out: &mut Vec<pipeline::Split>,
    ) {
        let base = scratch.len() as u32;
        if self.prepend_scheme == PrependScheme::Always
            && !piece.starts_with([' ', self.replacement])
        {
            scratch.push(self.replacement);
        }
        let mut split_start = base;
        for ch in piece.chars() {
            let ch = if ch == ' ' { self.replacement } else { ch };
            if self.split && ch == self.replacement {
                let pos = scratch.len() as u32;
                if pos > split_start {
                    out.push(pipeline::Split {
                        start: split_start,
                        end: pos,
                    });
                }
                split_start = pos;
            }
            scratch.push(ch);
        }
        let end = scratch.len() as u32;
        if end > split_start {
            out.push(pipeline::Split {
                start: split_start,
                end,
            });
        }
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
    fn rewrite_pieces(pretok: &Metaspace, pieces: &[&str]) -> Vec<String> {
        let mut scratch = String::new();
        let mut out = Vec::new();
        for piece in pieces {
            pretok.rewrite(piece, &mut scratch, &mut out);
        }
        out.iter().map(|s| scratch[s.range()].to_string()).collect()
    }

    /// Classic-path splits (Normalized referential) — the oracle for `rewrite`.
    fn legacy_pieces(pretok: &Metaspace, text: &str) -> Vec<String> {
        let mut pretokenized = PreTokenizedString::from(text);
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        pretokenized
            .get_splits(OffsetReferential::Normalized, OffsetType::Byte)
            .into_iter()
            .map(|(s, _, _)| s.to_string())
            .collect()
    }

    #[test]
    fn rewrite_matches_legacy() {
        let texts = [
            "Hey friend!",
            "Hey   friend!",
            " Hey",
            "▁already marked",
            "how▁are you",
            "no-spaces",
            "中文 text 123",
            "   ",
        ];
        for (scheme, split) in [
            (PrependScheme::Always, true),
            (PrependScheme::Always, false),
            (PrependScheme::Never, true),
            (PrependScheme::Never, false),
        ] {
            let pretok = Metaspace::new('▁', scheme, split);
            for text in texts {
                assert_eq!(
                    rewrite_pieces(&pretok, &[text]),
                    legacy_pieces(&pretok, text),
                    "{scheme:?} split={split} diverged on {text:?}",
                );
            }
        }
    }

    #[test]
    fn rewrite_prepends_per_piece() {
        // T5's shape: WhitespaceSplit runs first, so Metaspace sees one piece
        // per word and its only effect is the per-piece prepend.
        let pretok = Metaspace::new('▁', PrependScheme::Always, true);
        assert_eq!(
            rewrite_pieces(&pretok, &["Hey", "friend!"]),
            vec!["▁Hey", "▁friend!"],
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
}
