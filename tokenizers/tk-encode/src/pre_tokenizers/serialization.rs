//! The two pre-tokenizers whose serde cannot be a derive on the type.
//!
//! Nine of the eleven are declared with `impl_serde_type!` in their own file — the macro gives them
//! a `#[serde(tag = "type")]` envelope whose tag is *required* — and `SplitPattern` and
//! `PrependScheme` are plain derives next to their definitions. What is left needs a constructor:
//!
//! * `Metaspace`, because `replacement` is private (it keeps the derived `str_rep` in sync) and
//!   because reading one applies a backwards-compatibility rule that no derive can express;
//! * `Split`, because `search` and `fsm` are derived from `pattern` and building it can *fail*, on a
//!   pattern the regex backend rejects and the native FSM does not recognise.
//!
//! ## The `"type"` tag is required for every pre-tokenizer, and that is load-bearing
//!
//! This is where the pre-tokenizers differ, as a group, from the decoders: four decoders carry a
//! bare `#[serde(tag = "type")]`, which only *writes* the tag, so a tag-less object reaches them.
//! **No pre-tokenizer is in that position** — the nine spelled with `impl_serde_type!` get the
//! macro's `Def`-plus-shim requirement, and the two here spell it as a one-variant tag enum in a
//! mandatory field.
//!
//! It matters because `PreTokenizerWrapper`'s legacy fallback is an *untagged* enum: a variant that
//! is lenient about the tag will happily claim a tag-less object that should have been rejected, and
//! `pre_tokenizer_deserialization_no_type` is the test that catches it.
//!
//! The tag being a mandatory *field* on the way in and the first field on the way out is
//! byte-identical to what `#[serde(tag = "type")]` emits: `"type"` first, then declaration order.

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::metaspace::{Metaspace, PrependScheme};
use super::split::{Split, SplitPattern};
use crate::tokenizer::normalizer::SplitDelimiterBehavior;

// -------------------------------------------------------------------------------------------------
// Metaspace
// -------------------------------------------------------------------------------------------------

/// The legacy rule, reproduced exactly: an absent `prepend_scheme` means `Always`, **not** `Never`;
/// `add_prefix_space: false` is accepted only alongside an explicit `prepend_scheme: "never"` and is
/// otherwise a hard error. `str_rep` is read and thrown away. This is the same rule, word for word,
/// as the `Metaspace` *decoder*'s — the two have to agree about what a `tokenizer.json` means, and
/// both are load-bearing for ids, so neither gets "fixed".
#[derive(Deserialize)]
enum MetaspaceTag {
    Metaspace,
}

#[derive(Deserialize)]
struct MetaspaceIn {
    #[serde(rename = "type")]
    _type: MetaspaceTag,
    replacement: char,
    add_prefix_space: Option<bool>,
    #[serde(default = "default_prepend_scheme")]
    prepend_scheme: PrependScheme,
    split: Option<bool>,
    #[serde(rename = "str_rep")]
    _str_rep: Option<String>,
}

fn default_prepend_scheme() -> PrependScheme {
    PrependScheme::Always
}

/// `str_rep` is not written out, being derived from `replacement`. So the shape is `type`,
/// `replacement`, `prepend_scheme`, `split` — declaration order minus the derived field.
#[derive(Serialize)]
#[serde(tag = "type", rename = "Metaspace")]
struct MetaspaceOut<'a> {
    replacement: char,
    prepend_scheme: &'a PrependScheme,
    split: bool,
}

impl Serialize for Metaspace {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        MetaspaceOut {
            replacement: self.get_replacement(),
            prepend_scheme: &self.prepend_scheme,
            split: self.split,
        }
        .serialize(s)
    }
}

impl<'de> Deserialize<'de> for Metaspace {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let mut helper = MetaspaceIn::deserialize(d)?;
        if helper.add_prefix_space == Some(false) {
            if helper.prepend_scheme != PrependScheme::Never {
                return Err(D::Error::custom(
                    "add_prefix_space does not match declared prepend_scheme",
                ));
            }
            helper.prepend_scheme = PrependScheme::Never;
        }
        Ok(Metaspace::new(
            helper.replacement,
            helper.prepend_scheme,
            helper.split.unwrap_or(true),
        ))
    }
}

// -------------------------------------------------------------------------------------------------
// Split
// -------------------------------------------------------------------------------------------------

#[derive(Deserialize)]
enum SplitTag {
    Split,
}

#[derive(Deserialize)]
struct SplitIn {
    #[serde(rename = "type")]
    _type: SplitTag,
    pattern: SplitPattern,
    behavior: SplitDelimiterBehavior,
    invert: bool,
}

#[derive(Serialize)]
#[serde(tag = "type", rename = "Split")]
struct SplitOut<'a> {
    pattern: &'a SplitPattern,
    behavior: &'a SplitDelimiterBehavior,
    invert: bool,
}

impl Serialize for Split {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        SplitOut {
            pattern: &self.pattern,
            behavior: &self.behavior,
            invert: self.invert,
        }
        .serialize(s)
    }
}

impl<'de> Deserialize<'de> for Split {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let helper = SplitIn::deserialize(d)?;
        Split::new(helper.pattern, helper.behavior, helper.invert).map_err(D::Error::custom)
    }
}

// -------------------------------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pre_tokenizers::byte_level::ByteLevel;
    use crate::pre_tokenizers::punctuation::Punctuation;

    /// `PrependScheme`'s `Display` is spelled out by hand so the name survives a build with no serde
    /// in it. This is what stops it drifting from the `rename_all = "snake_case"` derive.
    #[test]
    fn display_matches_serde() {
        for scheme in [
            PrependScheme::First,
            PrependScheme::Never,
            PrependScheme::Always,
        ] {
            let via_serde = serde_json::to_string(&scheme).unwrap();
            assert_eq!(format!("\"{scheme}\""), via_serde);
        }
    }

    /// An absent `behavior` defaults to `Isolated`, which is `Punctuation::default()`.
    #[test]
    fn punctuation_deserialization() {
        let punctuation =
            serde_json::from_str::<Punctuation>(r#"{"type": "Punctuation"}"#).unwrap();
        assert_eq!(punctuation, Punctuation::default());
        assert_eq!(
            punctuation,
            Punctuation::new(SplitDelimiterBehavior::Isolated)
        );
    }

    /// And the required tag is what makes another pre-tokenizer's object fail rather than parse as a
    /// defaulted `Punctuation`.
    #[test]
    #[should_panic]
    fn punctuation_deserialization_erroneous() {
        let _punctuation =
            serde_json::from_str::<Punctuation>(r#"{"type": "WhitespaceSplit"}"#).unwrap();
    }

    /// `use_regex` was added after `ByteLevel` shipped, so an object without it has to keep loading
    /// with the regex on.
    #[test]
    fn byte_level_deserialization() {
        // Before use_regex
        let byte_level = serde_json::from_str::<ByteLevel>(
            r#"{"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false}"#,
        )
        .unwrap();
        assert!(byte_level.use_regex);

        // Loading works, new future BC test.
        let byte_level = serde_json::from_str::<ByteLevel>(
            r#"{"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": true}"#,
        )
        .unwrap();
        assert!(byte_level.use_regex);

        let byte_level = serde_json::from_str::<ByteLevel>(
            r#"{"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": false}"#,
        )
        .unwrap();
        assert!(!byte_level.use_regex);
    }

    /// The legacy `add_prefix_space` / `str_rep` rule.
    #[test]
    fn metaspace_serialization() {
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

    /// A config spelling its pattern as a string must also *deserialize* with no backend — the regex
    /// half of `split_serialization` below can only run once one is compiled.
    #[test]
    fn a_string_pattern_deserializes_with_no_backend() {
        let split_s =
            r#"{"type":"Split","pattern":{"String":"Hello"},"behavior":"Removed","invert":true}"#;
        let split = Split::new("Hello", SplitDelimiterBehavior::Removed, true).unwrap();
        assert_eq!(serde_json::from_str::<Split>(split_s).unwrap(), split);
        assert_eq!(serde_json::to_string(&split).unwrap(), split_s);
    }

    #[cfg(feature = "fancy-regex")] // needs a system-regex backend
    #[test]
    fn split_serialization() {
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
