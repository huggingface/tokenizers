//! The one pre-tokenizer whose serde cannot be a derive on the type.
//!
//! Nine of the ten are declared with `impl_serde_type!` in their own file — the macro gives them
//! a `#[serde(tag = "type")]` envelope whose tag is *required* — and `SplitPattern` is a plain
//! derive next to its definition. What is left needs a constructor:
//!
//! * `Split`, because `search` and `fsm` are derived from `pattern` and building it can *fail*, on a
//!   pattern the regex backend rejects and the native FSM does not recognise.
//!
//! ## The `"type"` tag is required for every pre-tokenizer, and that is load-bearing
//!
//! This is where the pre-tokenizers differ, as a group, from the decoders: four decoders carry a
//! bare `#[serde(tag = "type")]`, which only *writes* the tag, so a tag-less object reaches them.
//! **No pre-tokenizer is in that position** — the nine spelled with `impl_serde_type!` get the
//! macro's `Def`-plus-shim requirement, and the one here spells it as a one-variant tag enum in a
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

use super::split::{Split, SplitPattern};
use crate::tokenizer::normalizer::SplitDelimiterBehavior;

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
