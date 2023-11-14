use crate::tokenizer::{Decoder, PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior};
use serde::{Deserialize, Deserializer, Serialize};

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

#[derive(Debug, Clone, PartialEq, Serialize, Eq)]
/// Replaces all the whitespaces by the provided meta character and then
/// splits on this character
#[serde(tag = "type")]
pub struct Metaspace {
    replacement: char,
    pub add_prefix_space: bool,
    pub prepend_scheme: PrependScheme,
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
            pub add_prefix_space: bool,
            #[serde(default = "default_prepend_scheme_value")]
            pub prepend_scheme: PrependScheme,
            #[serde(skip, rename = "str_rep")]
            _str_rep: String,
        }

        let helper = MetaspaceHelper::deserialize(deserializer)?;
        let instance = Self::new_with_prepend_scheme(
            helper.replacement,
            helper.add_prefix_space,
            helper.prepend_scheme,
        );
        Ok(instance)
    }
}

impl Metaspace {
    pub fn new(replacement: char, add_prefix_space: bool) -> Self {
        Self::new_with_prepend_scheme(
            replacement,
            add_prefix_space,
            PrependScheme::Always, // always prepend for legacy purpose
        )
    }

    pub fn new_with_prepend_scheme(
        replacement: char,
        add_prefix_space: bool,
        prepend_scheme: PrependScheme,
    ) -> Self {
        Self {
            replacement,
            str_rep: replacement.to_string(),
            add_prefix_space,
            prepend_scheme,
        }
    }

    pub fn get_replacement(&self) -> char {
        self.replacement
    }

    pub fn set_replacement(&mut self, replacement: char) {
        self.replacement = replacement;
        self.str_rep = replacement.to_string();
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
        Self::new('▁', true)
    }
}

impl PreTokenizer for Metaspace {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        let mut first_split = true;

        pretokenized.split(|_, mut normalized| {
            normalized.replace(' ', &self.str_rep)?;
            if self.add_prefix_space && !normalized.get().starts_with(self.replacement) {
                if self.prepend_scheme == PrependScheme::Always {
                    normalized.prepend(&self.str_rep);
                } else if self.prepend_scheme == PrependScheme::First && first_split {
                    normalized.prepend(&self.str_rep);
                    first_split = false;
                }
            } else {
                first_split = false;
            }

            normalized.split(self.replacement, SplitDelimiterBehavior::MergedWithNext)
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
                            if i == 0 && self.add_prefix_space {
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
        let metaspace = Metaspace::new('_', true);
        let metaspace_s = r#"{"type":"Metaspace","replacement":"_","add_prefix_space":true,"prepend_scheme":"always"}"#;
        assert_eq!(serde_json::to_string(&metaspace).unwrap(), metaspace_s);
        assert_eq!(
            serde_json::from_str::<Metaspace>(metaspace_s).unwrap(),
            metaspace
        );

        // Also check it can deserialize previous versions
        let metaspace = Metaspace::new('_', true);
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
        let pretok = Metaspace::new('▁', true);
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
        let pretok = Metaspace::new('▁', true);
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
        assert_eq!(
            Metaspace::new('▁', true),
            Metaspace::new_with_prepend_scheme('▁', true, PrependScheme::Always)
        );

        let mut pretok = Metaspace::new('▁', true);
        pretok.set_prepend_scheme(PrependScheme::Always);
        assert_eq!(
            pretok,
            Metaspace::new_with_prepend_scheme('▁', true, PrependScheme::Always)
        );

        pretok.set_prepend_scheme(PrependScheme::Never);
        assert_eq!(
            pretok,
            Metaspace::new_with_prepend_scheme('▁', true, PrependScheme::Never)
        );

        pretok.set_prepend_scheme(PrependScheme::First);
        assert_eq!(
            pretok,
            Metaspace::new_with_prepend_scheme('▁', true, PrependScheme::First)
        );

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
                ("▁Hey", (0, 6)),
                ("▁my", (6, 11)),
                ("▁friend", (11, 20)),
                ("▁", (20, 23)),
                ("<s>", (23, 26)),
                ("how", (26, 29)),
                ("▁are", (29, 35)),
                ("▁you", (35, 41))
            ]
        );
        pretok.set_prepend_scheme(PrependScheme::Always);
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

        pretok.set_prepend_scheme(PrependScheme::First);
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
            vec![
                ("▁Hey", (0, 6)),
                ("▁", (6, 9)),
                ("<s>", (9, 12)),
                ("how", (12, 15))
            ]
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
                ("▁Hey", (0, 6)),
                ("▁", (6, 9)),
                ("<s>", (9, 12)),
                ("how", (12, 15)),
                ("▁", (15, 18)),
                ("<s>", (18, 21)),
                ("are", (21, 24)),
                ("▁", (24, 27)),
                ("<s>", (27, 30)),
                ("▁you", (30, 36))
            ]
        );
    }
    #[test]
    fn decode() {
        let decoder = Metaspace::new('▁', true);
        let res = decoder
            .decode_chain(vec!["▁Hey".into(), "▁friend!".into()])
            .unwrap();
        assert_eq!(res, vec!["Hey", " friend!"])
    }
}
