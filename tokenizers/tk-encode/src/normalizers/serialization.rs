//! serde for every normalizer: `tk-encode` defines the thirteen normalizer types and serializes
//! none of them.
//!
//! Same orphan-rule reason as [`crate::decoders::mirror`], which is the worked example this file
//! repeats: `Serialize` and `Deserialize` are foreign traits and `NFD`/`BertNormalizer`/`Replace`
//! and the rest are foreign types, so this crate cannot implement one for the other. What it *can*
//! do is define a local mirror of the JSON shape and convert — which is what
//! `#[serde(with = "...")]` on the wrapper's variants asks for.
//!
//! `Precompiled` is the one normalizer with no mirror here. It is `spm_precompiled::Precompiled`,
//! re-exported: the serde lives in that crate, not in ours, so there is nothing to move. Its
//! base64 charsmap also only decodes through the *string* deserializer, which is why
//! `NormalizerWrapper` re-serialises the value and calls `from_str` for that one variant.
//! `MetaspaceNormalizer` has no mirror either, for the opposite reason: it never appears in a
//! `tokenizer.json`. It is half of what a `Metaspace` *pre-tokenizer* lowers into, so it is built
//! by `to_normalizer_and_split` and never read from a config.
//!
//! ## Two mirror shapes, and how to pick
//!
//! * **`remote`** — serde drives the foreign type directly. Only usable when every field is `pub`
//!   *and* the type is not `#[non_exhaustive]`: the generated code builds the type with a struct
//!   literal, which `#[non_exhaustive]` forbids across a crate boundary. `Prepend` is the one
//!   normalizer that qualifies.
//! * **an explicit `mod` with `serialize`/`deserialize`** — a local struct carrying *the same serde
//!   attributes the type used to carry*, converted through the type's own constructor. `Strip` and
//!   `BertNormalizer` use this because they are `#[non_exhaustive]`; `Replace` because its
//!   `pattern` is private behind a getter and its constructor can fail.
//!
//! The eight fieldless normalizers are neither: they get [`unit_mirror`] below, which is the
//! `mirror::fuse` / `mirror::byte_fallback` shape from the decoders, written once.
//!
//! ## Whether the `"type"` tag is required is per-normalizer, and it is load-bearing
//!
//! `NormalizerWrapper`'s legacy fallback is an *untagged* enum, so a variant that is lenient about
//! the tag will happily claim a tag-less object that should have been rejected. The split here is
//! cleaner than it was for the decoders, because it falls exactly along which macro the type used:
//!
//! * **required** for the eight fieldless normalizers — `NFC`, `NFD`, `NFKC`, `NFKD`, `Nmt`,
//!   `StripAccents`, `Lowercase` and `ByteLevel`. Each was written with `impl_serde_type!`, whose
//!   unit-struct arm hand-writes a `Helper` whose *only* field is the tag. Nothing else would stop
//!   `{}` from deserializing as an `NFD`, and `normalizer_serialization` is the test that catches
//!   it: a `Sequence` containing `{}` has to fail. Their mirrors reproduce the requirement with a
//!   one-variant tag enum.
//! * **optional** for `Strip`, `BertNormalizer`, `Prepend` and `Replace`, which carried a bare
//!   `#[serde(tag = "type")]`. As `crate::macros` spells out, that attribute does **not** make the
//!   tag required — and it is worth being precise about how lenient it really is, because the
//!   `Bert` note below turns on it: serde *ignores the tag field entirely* on the way in. A wrong
//!   tag value is accepted, an absent one is accepted, and only the ordinary required fields do any
//!   rejecting. These four mirrors carry the identical attribute, so the codegen — and therefore
//!   the accepted input — is unchanged. `post_processor_deserialization_no_type` is the test that
//!   depends on it: `{"strip_left":false,"strip_right":true}` must still load as a `Strip`.
//!
//! ## The `Bert` tag is spelled two ways, and both have to keep working
//!
//! Worth writing down because it looks like a bug in passing and is not one to fix here.
//! `#[serde(tag = "type")]` on a struct called `BertNormalizer` writes `"type":"BertNormalizer"`,
//! which is what every real config on disk says. But `NormalizerWrapper`'s `EnumType` spells that
//! variant `Bert`. So a real bert config does *not* match the tagged path at all — `EnumType`
//! rejects the unknown variant, the outer untagged helper falls through to `Legacy`, and the
//! `NormalizerUntagged` enum matches it on its fields. The tagged `EnumType::Bert` arm is reached
//! only by a config that literally says `"type":"Bert"`, and it works only because that arm
//! re-inserts `"Bert"` as the tag and the mirror ignores it. Both spellings load today, so both
//! spellings load after this move: [`bert`] keeps the lenient bare attribute rather than gaining a
//! one-variant tag enum, which would have quietly broken the `"type":"Bert"` spelling *and* sent
//! every real bert config down a failing path.

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use tk_encode::normalizers::bert::BertNormalizer;
use tk_encode::normalizers::byte_level::ByteLevel;
use tk_encode::normalizers::prepend::Prepend;
use tk_encode::normalizers::replace::Replace;
use tk_encode::normalizers::strip::{Strip, StripAccents};
use tk_encode::normalizers::unicode::{NFC, NFD, NFKC, NFKD, Nmt};
use tk_encode::normalizers::utils::Lowercase;
use tk_encode::utils::search::ReplacePattern;

use crate::decoders::mirror::ReplacePatternDef;

// ---------------------------------------------------------------------------------------------
// The eight fieldless normalizers
// ---------------------------------------------------------------------------------------------

/// One mirror for a normalizer with no fields, where the tag is the only thing on the wire.
///
/// This is `mirror::fuse` from the decoders, except that eight normalizers need it rather than two,
/// so it is a macro instead of eight copies of the same twenty lines. What it expands to is exactly
/// what `impl_serde_type!`'s unit-struct arm used to expand to on the `tk-encode` side: a
/// one-variant enum for the tag, a struct whose only field is that tag, and nothing else. The
/// one-variant enum is what makes the tag *required*; see the module docs.
///
/// The type name doubles as the value, which every one of the eight allows because they are all
/// unit structs, and as the tag's variant name, which is what serialises it back to the same
/// spelling the type used to write itself: `NFD` becomes `{"type":"NFD"}`.
macro_rules! unit_mirror {
    ($mod_name:ident, $ty:ident) => {
        paste::paste! {
            #[derive(Serialize, Deserialize)]
            enum [<$ty Tag>] {
                $ty,
            }

            #[derive(Serialize, Deserialize)]
            struct [<$ty Mirror>] {
                #[serde(rename = "type")]
                _type: [<$ty Tag>],
            }

            pub mod $mod_name {
                use super::*;

                pub fn serialize<S: Serializer>(_v: &$ty, s: S) -> Result<S::Ok, S::Error> {
                    [<$ty Mirror>] {
                        _type: [<$ty Tag>]::$ty,
                    }
                    .serialize(s)
                }

                pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<$ty, D::Error> {
                    [<$ty Mirror>]::deserialize(d)?;
                    Ok($ty)
                }
            }
        }
    };
}

unit_mirror!(nfd, NFD);
unit_mirror!(nfkd, NFKD);
unit_mirror!(nfc, NFC);
unit_mirror!(nfkc, NFKC);
unit_mirror!(nmt, Nmt);
unit_mirror!(strip_accents, StripAccents);
unit_mirror!(lowercase, Lowercase);
unit_mirror!(byte_level, ByteLevel);

// ---------------------------------------------------------------------------------------------
// Strip
// ---------------------------------------------------------------------------------------------

/// `#[non_exhaustive]`, so `remote` cannot drive it — hence the explicit module through
/// `Strip::new`. Both fields are required, which is the *only* thing that rejects a tag-less
/// object here: the tag itself is ignored on the way in, exactly as it was before the move.
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename = "Strip")]
struct StripMirror {
    strip_left: bool,
    strip_right: bool,
}

pub mod strip {
    use super::*;

    pub fn serialize<S: Serializer>(v: &Strip, s: S) -> Result<S::Ok, S::Error> {
        StripMirror {
            strip_left: v.strip_left,
            strip_right: v.strip_right,
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Strip, D::Error> {
        let m = StripMirror::deserialize(d)?;
        Ok(Strip::new(m.strip_left, m.strip_right))
    }
}

// ---------------------------------------------------------------------------------------------
// BertNormalizer
// ---------------------------------------------------------------------------------------------

/// `#[non_exhaustive]` again, so again an explicit module through `BertNormalizer::new`.
///
/// All four fields are required — including `strip_accents`, which is an `Option<bool>` but has no
/// `#[serde(default)]`, so a config has to spell it, if only as `null`. That is not an oversight to
/// tidy: `null` and *absent* would mean the same thing to the type (`None`, meaning "follow
/// `lowercase`"), but they do not mean the same thing to serde, and `tk-serialize`'s slim reader
/// rejects the absent case with a message of its own. The field order is the declaration order of
/// `BertNormalizer`, which `tests/serialization.rs` pins as a string.
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename = "BertNormalizer")]
struct BertNormalizerMirror {
    clean_text: bool,
    handle_chinese_chars: bool,
    strip_accents: Option<bool>,
    lowercase: bool,
}

pub mod bert {
    use super::*;

    pub fn serialize<S: Serializer>(v: &BertNormalizer, s: S) -> Result<S::Ok, S::Error> {
        BertNormalizerMirror {
            clean_text: v.clean_text,
            handle_chinese_chars: v.handle_chinese_chars,
            strip_accents: v.strip_accents,
            lowercase: v.lowercase,
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<BertNormalizer, D::Error> {
        let m = BertNormalizerMirror::deserialize(d)?;
        Ok(BertNormalizer::new(
            m.clean_text,
            m.handle_chinese_chars,
            m.strip_accents,
            m.lowercase,
        ))
    }
}

// ---------------------------------------------------------------------------------------------
// Prepend
// ---------------------------------------------------------------------------------------------

/// The one normalizer `remote` can drive: a single `pub` field, no invariant to uphold and no
/// `#[non_exhaustive]`, so serde builds it with a struct literal and no hand-written conversion is
/// needed.
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename = "Prepend", remote = "Prepend")]
pub struct PrependDef {
    pub prepend: String,
}

// ---------------------------------------------------------------------------------------------
// Replace
// ---------------------------------------------------------------------------------------------

/// `Replace` keeps a compiled matcher derived from `pattern`, so it can only be built through its
/// constructor -- which can fail, on a pattern the regex backend rejects. That is what the
/// `ReplaceDeserializer` + `TryFrom` pair on the `tk-encode` side did, and this module is that pair
/// with the serde moved across.
///
/// The `search` field's `#[serde(skip)]` disappears rather than moving: a mirror only names the
/// fields that are on the wire, and a derived matcher never was.
///
/// [`ReplacePatternDef`] is reused from the decoders' mirror rather than declared again -- one
/// pattern spelling, one place. Note this is the *normalizer* `Replace`, a different type from the
/// decoder of the same name, but the on-disk shape is the same two fields.
#[derive(Deserialize)]
#[serde(tag = "type", rename = "Replace")]
struct ReplaceIn {
    #[serde(with = "ReplacePatternDef")]
    pattern: ReplacePattern,
    content: String,
}

#[derive(Serialize)]
#[serde(tag = "type", rename = "Replace")]
struct ReplaceOut<'a> {
    #[serde(with = "ReplacePatternDef")]
    pattern: &'a ReplacePattern,
    content: &'a str,
}

pub mod replace {
    use super::*;

    pub fn serialize<S: Serializer>(v: &Replace, s: S) -> Result<S::Ok, S::Error> {
        ReplaceOut {
            pattern: v.pattern(),
            content: &v.content,
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Replace, D::Error> {
        let m = ReplaceIn::deserialize(d)?;
        Replace::new(m.pattern, m.content).map_err(D::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalizers::NormalizerWrapper;

    use tk_encode::normalizers::precompiled::Precompiled;
    use tk_encode::{NormalizedString, Normalizer, pipeline};

    /// Deserialize one normalizer the way a `tokenizer.json` does, and hand back the inner value.
    ///
    /// The wrapper rather than the leaf type is the unit under test throughout this module, because
    /// the leaf types have no serde of their own any more. That is a feature, not a compromise: it
    /// exercises the tag routing at the same time as the shape.
    fn from_json(json: &str) -> NormalizerWrapper {
        serde_json::from_str(json).unwrap()
    }

    /// Moved from `tk-encode`'s `normalizers::replace` with the serde it exercises. Both spellings
    /// of a pattern round-trip, byte for byte, to the same JSON they always did.
    #[test]
    #[cfg(feature = "fancy-regex")] // the regex half of this needs a system-regex backend
    fn replace_serialization() {
        let replace = Replace::new("Hello", "Hey").unwrap();
        let replace_s = r#"{"type":"Replace","pattern":{"String":"Hello"},"content":"Hey"}"#;
        let wrapped = NormalizerWrapper::Replace(replace.clone());
        assert_eq!(serde_json::to_string(&wrapped).unwrap(), replace_s);
        match from_json(replace_s) {
            NormalizerWrapper::Replace(back) => assert_eq!(back, replace),
            other => panic!("Replace wrapped with incorrect variant: {other:?}"),
        }

        let replace = Replace::new(ReplacePattern::Regex(r"\s+".into()), ' ').unwrap();
        let replace_s = r#"{"type":"Replace","pattern":{"Regex":"\\s+"},"content":" "}"#;
        let wrapped = NormalizerWrapper::Replace(replace.clone());
        assert_eq!(serde_json::to_string(&wrapped).unwrap(), replace_s);
        match from_json(replace_s) {
            NormalizerWrapper::Replace(back) => assert_eq!(back, replace),
            other => panic!("Replace wrapped with incorrect variant: {other:?}"),
        }
    }

    /// Also moved from `tk-encode`: a config spelling its pattern as a string must *deserialize*
    /// with no regex backend compiled in, which is why this one carries no `fancy-regex` gate while
    /// `replace_serialization` above does.
    #[test]
    fn a_string_pattern_deserializes_with_no_backend() {
        let replace_s = r#"{"type":"Replace","pattern":{"String":"Hello"},"content":"Hey"}"#;
        let replace = Replace::new("Hello", "Hey").unwrap();
        match from_json(replace_s) {
            NormalizerWrapper::Replace(back) => assert_eq!(back, replace),
            other => panic!("Replace wrapped with incorrect variant: {other:?}"),
        }
        assert_eq!(
            serde_json::to_string(&NormalizerWrapper::Replace(replace)).unwrap(),
            replace_s
        );
    }

    /// The eight fieldless normalizers reject an object with no tag, which is the whole reason
    /// their mirrors carry a one-variant tag enum instead of nothing at all. Were they lenient,
    /// `NormalizerUntagged` would match `{}` as the first fieldless variant it tried.
    #[test]
    fn a_fieldless_normalizer_requires_its_tag() {
        for tag in [
            "NFD",
            "NFKD",
            "NFC",
            "NFKC",
            "Nmt",
            "StripAccents",
            "Lowercase",
            "ByteLevel",
        ] {
            let json = format!(r#"{{"type":"{tag}"}}"#);
            // Round-trips to exactly the object it came from, tag and all.
            assert_eq!(serde_json::to_string(&from_json(&json)).unwrap(), json);
        }

        let err = serde_json::from_str::<NormalizerWrapper>("{}").unwrap_err();
        assert_eq!(
            err.to_string(),
            "data did not match any variant of untagged enum NormalizerUntagged"
        );

        // And one is not interchangeable with another: the tag enum has a single variant, so the
        // wrong spelling is an unknown variant rather than an ignored field.
        assert!(mirror_nfd(r#"{"type":"NFC"}"#).is_err());
        assert!(mirror_nfd(r#"{"type":"NFD"}"#).is_ok());
        assert!(mirror_nfd("{}").is_err());
    }

    fn mirror_nfd(json: &str) -> Result<tk_encode::normalizers::unicode::NFD, serde_json::Error> {
        let value: serde_json::Value = serde_json::from_str(json).unwrap();
        nfd::deserialize(value)
    }

    /// `BertNormalizer` is written `"type":"BertNormalizer"` by every real config, but
    /// `NormalizerWrapper`'s `EnumType` spells the variant `Bert`. Both have to keep loading; see
    /// the module docs for which path each one takes. This is the test that stops a future tidy-up
    /// from giving `bert` a required tag and quietly breaking one of them.
    #[test]
    fn bert_loads_under_both_tag_spellings() {
        let fields = r#""clean_text":true,"handle_chinese_chars":true,"strip_accents":null,"lowercase":true"#;

        // The spelling on disk. Goes through the *untagged* legacy fallback, because `EnumType`
        // has no `BertNormalizer` variant to match.
        let on_disk = format!(r#"{{"type":"BertNormalizer",{fields}}}"#);
        assert!(matches!(
            from_json(&on_disk),
            NormalizerWrapper::BertNormalizer(_)
        ));
        // Whichever path it took, it writes itself back out the way configs spell it.
        assert_eq!(
            serde_json::to_string(&from_json(&on_disk)).unwrap(),
            on_disk
        );

        // The spelling `EnumType` matches. Goes through the tagged path, which re-inserts `"Bert"`
        // as the tag -- accepted only because a bare `#[serde(tag = "type")]` ignores it.
        let enum_spelling = format!(r#"{{"type":"Bert",{fields}}}"#);
        assert!(matches!(
            from_json(&enum_spelling),
            NormalizerWrapper::BertNormalizer(_)
        ));
    }

    /// Moved from `tk-encode`'s `normalizers::precompiled`: the only charsmap to test against is
    /// one read out of a `tokenizer.json`, so the test needs serde_json and therefore lives here.
    fn albert_precompiled() -> Precompiled {
        let json = std::fs::read_to_string("../data/albert-base-v1-tokenizer.json").unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let precompiled = value["normalizer"]["normalizers"]
            .as_array()
            .unwrap()
            .iter()
            .find(|n| n["type"] == "Precompiled")
            .unwrap();
        // Precompiled can't deserialize through serde_json::Value (the base64
        // charsmap only decodes via the string deserializer) — same dance as
        // NormalizerWrapper's Deserialize impl
        serde_json::from_str(&serde_json::to_string(precompiled).unwrap()).unwrap()
    }

    #[test]
    fn pipeline_precompiled_matches_legacy() {
        let n = albert_precompiled();
        let mut any_modified = false;
        for input in &[
            "™\x1eg",
            "ＫＡＤＯＫＡＷＡ",
            "１２３",
            "…",
            "\u{fb01}",
            "e\u{0301}",
            "㍿",
            "abc def",
            "",
        ] {
            let mut ns = NormalizedString::from(*input);
            Normalizer::normalize(&n, &mut ns).unwrap(); // legacy oracle
            any_modified |= ns.get() != *input;
            assert_eq!(
                ns.get(),
                &*pipeline::Normalizer::normalize(&n, input).unwrap(),
                "pipeline output diverges from legacy for {input:?}"
            );
        }
        // Guard against the oracle silently becoming a no-op on these inputs
        assert!(any_modified);
    }
}
