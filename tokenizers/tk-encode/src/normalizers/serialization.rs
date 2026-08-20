//! The one normalizer whose serde cannot be a derive on the type.
//!
//! Everything else carries its own, next to the struct:
//!
//! * the eight field-less normalizers — `NFC`, `NFD`, `NFKC`, `NFKD`, `Nmt`, `StripAccents`,
//!   `Lowercase`, `ByteLevel` — are declared with `impl_serde_type!`, whose unit-struct arm hand-writes
//!   a `Helper` whose *only* field is the tag. That is what makes the tag **required**, and nothing
//!   else would stop `{}` from deserializing as an `NFD`;
//! * `Strip`, `BertNormalizer` and `Prepend` are plain derives with a bare `#[serde(tag = "type")]`,
//!   which does not require the tag at all — serde ignores its value and its absence on the way in,
//!   and only the ordinary required fields do any rejecting. `post_processor_deserialization_no_type`
//!   depends on that: `{"strip_left":false,"strip_right":true}` must still load as a `Strip`;
//! * `Precompiled` is `spm_precompiled::Precompiled`, re-exported, so its serde lives in that crate.
//!   Its base64 charsmap also only decodes through the *string* deserializer, which is why
//!   `NormalizerWrapper` re-serialises the value and calls `from_str` for that one variant;
//! * `MetaspaceNormalizer` has none at all, and needs none: it never appears in a `tokenizer.json`.
//!   It is half of what a `Metaspace` *pre-tokenizer* lowers into, built by `to_normalizer_and_split`
//!   and never read from a config.
//!
//! Whether the tag is required is decided per normalizer and it is load-bearing, because
//! `NormalizerWrapper`'s legacy fallback is an *untagged* enum: a lenient variant will claim a
//! tag-less object that should have been rejected. The split above is exactly the split between the
//! two spellings, and `a_fieldless_normalizer_requires_its_tag` is the test that pins it.

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::replace::Replace;
use crate::utils::search::ReplacePattern;

/// `Replace` keeps a compiled matcher derived from `pattern`, so it can only be built through its
/// constructor -- which can fail, on a pattern the regex backend rejects. The `search` field is
/// derived and never reaches the wire.
///
/// Note this is the *normalizer* `Replace`, a different type from the decoder of the same name, but
/// the on-disk shape is the same two fields.
#[derive(Deserialize)]
#[serde(tag = "type", rename = "Replace")]
struct ReplaceIn {
    pattern: ReplacePattern,
    content: String,
}

#[derive(Serialize)]
#[serde(tag = "type", rename = "Replace")]
struct ReplaceOut<'a> {
    pattern: &'a ReplacePattern,
    content: &'a str,
}

impl Serialize for Replace {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        ReplaceOut {
            pattern: self.pattern(),
            content: &self.content,
        }
        .serialize(s)
    }
}

impl<'de> Deserialize<'de> for Replace {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let m = ReplaceIn::deserialize(d)?;
        Replace::new(m.pattern, m.content).map_err(D::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both spellings of a pattern round-trip, byte for byte, to the same JSON they always did.
    #[test]
    #[cfg(feature = "fancy-regex")] // the regex half of this needs a system-regex backend
    fn replace_serialization() {
        let replace = Replace::new("Hello", "Hey").unwrap();
        let replace_s = r#"{"type":"Replace","pattern":{"String":"Hello"},"content":"Hey"}"#;
        assert_eq!(serde_json::to_string(&replace).unwrap(), replace_s);
        assert_eq!(serde_json::from_str::<Replace>(replace_s).unwrap(), replace);

        let replace = Replace::new(ReplacePattern::Regex(r"\s+".into()), ' ').unwrap();
        let replace_s = r#"{"type":"Replace","pattern":{"Regex":"\\s+"},"content":" "}"#;
        assert_eq!(serde_json::to_string(&replace).unwrap(), replace_s);
        assert_eq!(serde_json::from_str::<Replace>(replace_s).unwrap(), replace);
    }

    /// A config spelling its pattern as a string must *deserialize* with no regex backend compiled
    /// in, which is why this one carries no `fancy-regex` gate while `replace_serialization` does.
    #[test]
    fn a_string_pattern_deserializes_with_no_backend() {
        let replace_s = r#"{"type":"Replace","pattern":{"String":"Hello"},"content":"Hey"}"#;
        let replace = Replace::new("Hello", "Hey").unwrap();
        assert_eq!(serde_json::from_str::<Replace>(replace_s).unwrap(), replace);
        assert_eq!(serde_json::to_string(&replace).unwrap(), replace_s);
    }
}
