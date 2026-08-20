use crate::normalizers::metaspace::MetaspaceNormalizer;
use crate::pre_tokenizers::split::Split;
use crate::tokenizer::{PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior};

/// Enum representing options for the metaspace prepending scheme.
///
/// The JSON spelling is `snake_case`: `"first"` / `"never"` / `"always"`. `decoders::metaspace`
/// re-exports this very type, so the decoder and the pre-tokenizer read and write it identically.
#[derive(Debug, Clone, PartialEq, Eq, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
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
        // Spelled out rather than handed to the serializer, so the name survives a build with no
        // serde in it. These must stay identical to the `rename_all = "snake_case"` spelling above;
        // `display_matches_serde` pins that.
        f.write_str(match self {
            Self::First => "first",
            Self::Never => "never",
            Self::Always => "always",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Replaces all the whitespaces by the provided meta character and then
/// splits on this character
///
/// The JSON shape -- including the backwards-compatible `add_prefix_space` / `str_rep` reading,
/// whose rule decides ids and so must not be "fixed" -- is in [`super::serialization`]. `str_rep` is
/// derived from `replacement` and never written out; that is why `replacement` is private, and why
/// reading one has to go through [`Metaspace::new`] rather than a struct literal.
pub struct Metaspace {
    replacement: char,
    pub prepend_scheme: PrependScheme,
    pub split: bool,
    str_rep: String,
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

/// The two pipeline steps a `Metaspace` pre-tokenizer decomposes into:
///   - a [`MetaspaceNormalizer`], writing the `▁` delimiters into the text,
///   - a [`Split`] on that delimiter, cutting the text into words.
///
/// [`crate::normalizers::metaspace`] explains what those delimiters are for, and why the pipeline
/// wants the two steps apart.
///
/// `None` for a `Metaspace` whose settings cannot be reproduced by a normalizer plus a `Split`. The
/// caller then rejects the pre-tokenizer, leaving the model out of the pipeline instead of quietly
/// encoding it differently.
///
// `pub` because the shape-matching half — "is this pre-tokenizer a `Metaspace`, or a
// `WhitespaceSplit` followed by one?" — is written against `PreTokenizerWrapper` and so lives in
// `tk-convert`. The *rule* below is the part that decides ids, and it stays single-sourced
// here; the slim reader's `read_metaspace` calls the same one.
pub fn normalizer_and_split(
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
    use regex::Regex;

    use super::*;
    use crate::{OffsetReferential, OffsetType};

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
}
