//! serde for every pre-tokenizer: `tk-encode` defines the twelve pre-tokenizer types and
//! serializes none of them.
//!
//! Same orphan-rule story as [`crate::decoders::mirror`], which is the worked example this file
//! repeats: `Serialize`/`Deserialize` are foreign traits, `Whitespace`/`Metaspace`/`Split` and the
//! rest are foreign types, so this crate can implement neither for the other. It declares a local
//! mirror of the JSON shape and converts, and `#[serde(with = "...")]` on the wrapper's variants is
//! what asks for that conversion.
//!
//! The mirrors are the authority on the on-disk shape, backwards-compatible spellings included:
//! `Metaspace`'s legacy `add_prefix_space` and its dead `str_rep`, `ByteLevel`'s defaulted
//! `use_regex`, `Punctuation`'s defaulted `behavior`, `FixedLength`'s defaulted `length`. The
//! runtime crate no longer knows or cares how a pre-tokenizer is written down.
//!
//! ## Two mirror shapes, and how to pick
//!
//! * **`remote`** — serde drives the foreign type directly. Only usable when every field is `pub`
//!   *and* the type is not `#[non_exhaustive]`, because the generated code builds the type with a
//!   struct literal, which `#[non_exhaustive]` forbids across a crate boundary. Here that is only
//!   the two *field* types, [`PrependSchemeDef`] and [`SplitPatternDef`]: plain enums with no
//!   invariants, named from other mirrors rather than from a wrapper variant.
//! * **an explicit `mod` with `serialize`/`deserialize`** — a local struct carrying the same serde
//!   attributes the type used to carry, converted through the type's own constructor. Every
//!   pre-tokenizer proper uses this, and for whichever of these reasons applies: the tag has to be
//!   *required* (see below, and it is all of them), the type is `#[non_exhaustive]`
//!   (`CharDelimiterSplit`, `Digits`, `ByteLevel`), or a field is private and derived
//!   (`Metaspace`'s `str_rep`, `Split`'s `search` and `fsm`).
//!
//! ## The `"type"` tag is required for every pre-tokenizer, and that is load-bearing
//!
//! This is the one place where the pre-tokenizers differ, as a group, from the decoders. Four
//! decoders carried a bare `#[serde(tag = "type")]` derive, which only *writes* the tag and does not
//! require it on the way in, so their mirrors keep an optional tag. **No pre-tokenizer is in that
//! position.** Every one of them arrived at its serde by one of three routes, and all three make the
//! tag mandatory:
//!
//!   * `#[macro_rules_attribute(impl_serde_type!)]` — nine of them (`BertPreTokenizer`,
//!     `Whitespace`, `WhitespaceSplit`, `UnicodeScripts`, `CharDelimiterSplit`, `Digits`,
//!     `FixedLength`, `Punctuation`, `ByteLevel`). As [`crate::macros`] spells out, that macro's
//!     `Def` remote plus its `Deserializer` shim exist *precisely* so that a missing `"type"` is an
//!     error rather than a silently-accepted bare struct. Its unit-struct arm does the same thing
//!     with a `Helper` whose only field is the tag.
//!   * a hand-written `Deserialize` whose helper opened with a one-variant tag enum — `Metaspace`
//!     and `Split`.
//!   * `Sequence`, which is this crate's own type and still calls `impl_serde_type!` directly.
//!
//! So every mirror below requires the tag, spelled the way `mirror::fuse` and
//! `mirror::byte_fallback` spell it in the decoders: a one-variant tag enum as a mandatory field
//! rather than `#[serde(tag = "...")]`. It matters because [`PreTokenizerWrapper`]'s legacy fallback
//! is an *untagged* enum: a variant that is lenient about the tag will happily claim a tag-less
//! object that should have been rejected, and `pre_tokenizer_deserialization_no_type` is the test
//! that catches it.
//!
//! Note the tag is a mandatory *field* on the way in and the first field on the way out, which is
//! byte-identical to what `#[serde(tag = "type")]` emits — `"type"` first, then declaration order.
//!
//! ## What the tag is spelled
//!
//! The mirrors spell the tag the way the *type* did, which for four of them is not the name
//! `PreTokenizerWrapper` uses for the variant: `CharDelimiterSplit` (variant `Delimiter`),
//! `BertPreTokenizer`, `WhitespaceSplit` and `UnicodeScripts`. `impl_serde_type!` derived the tag
//! from the struct name, so `{"type":"CharDelimiterSplit"}` is what is on disk and what has to keep
//! loading — through the untagged legacy fallback, since `EnumType` has no such variant. Renaming
//! any of these would silently change which configs parse.
//!
//! [`PreTokenizerWrapper`]: super::PreTokenizerWrapper

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use tk_encode::pre_tokenizers::bert::BertPreTokenizer;
use tk_encode::pre_tokenizers::byte_level::ByteLevel;
use tk_encode::pre_tokenizers::delimiter::CharDelimiterSplit;
use tk_encode::pre_tokenizers::digits::Digits;
use tk_encode::pre_tokenizers::fixed_length::FixedLength;
use tk_encode::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tk_encode::pre_tokenizers::punctuation::Punctuation;
use tk_encode::pre_tokenizers::split::{Split, SplitPattern};
use tk_encode::pre_tokenizers::unicode_scripts::UnicodeScripts;
use tk_encode::pre_tokenizers::whitespace::{Whitespace, WhitespaceSplit};
use tk_encode::tokenizer::normalizer::SplitDelimiterBehavior;

use crate::mirror::SplitDelimiterBehaviorDef;

// -------------------------------------------------------------------------------------------------
// Field types
// -------------------------------------------------------------------------------------------------

/// `PrependScheme` is a plain three-variant enum with no invariants, so `remote` drives it directly.
///
/// It lives here, in the pre-tokenizers, rather than in [`crate::mirror`] with the other shared
/// types, because that is where `tk-encode` puts the type itself: `pre_tokenizers::metaspace`
/// defines it and `decoders::metaspace` re-exports it. `decoders::mirror::metaspace` names this one
/// too, for exactly that reason — there is only one `PrependScheme`, so there is only one mirror of
/// it, and it sits on the side that owns the definition.
///
/// `tk-encode` spells out its `Display` by hand and `display_matches_serde` (below) keeps the two
/// from drifting; that test moved here with the serde half it asserts.
#[derive(Serialize, Deserialize)]
#[serde(remote = "PrependScheme", rename_all = "snake_case")]
pub enum PrependSchemeDef {
    First,
    Never,
    Always,
}

/// Externally tagged — `{"String":"..."}` / `{"Regex":"..."}` — which is what the bare derive on
/// `SplitPattern` produced, and what is on disk. Both variants are one public `String`, so `remote`
/// drives it directly.
#[derive(Serialize, Deserialize)]
#[serde(remote = "SplitPattern")]
pub enum SplitPatternDef {
    String(String),
    Regex(String),
}

// -------------------------------------------------------------------------------------------------
// BertPreTokenizer
// -------------------------------------------------------------------------------------------------

/// No fields, so the tag is the only thing on the wire — and the only thing that can make `{}` fail.
/// See the module docs.
#[derive(Serialize, Deserialize)]
enum BertPreTokenizerTag {
    BertPreTokenizer,
}

#[derive(Serialize, Deserialize)]
struct BertPreTokenizerMirror {
    #[serde(rename = "type")]
    _type: BertPreTokenizerTag,
}

pub mod bert {
    use super::*;

    pub fn serialize<S: Serializer>(_v: &BertPreTokenizer, s: S) -> Result<S::Ok, S::Error> {
        BertPreTokenizerMirror {
            _type: BertPreTokenizerTag::BertPreTokenizer,
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<BertPreTokenizer, D::Error> {
        BertPreTokenizerMirror::deserialize(d)?;
        Ok(BertPreTokenizer)
    }
}

// -------------------------------------------------------------------------------------------------
// Whitespace
// -------------------------------------------------------------------------------------------------

/// Another field-less one. `tests/serialization.rs` in the umbrella crate asserts that a
/// `BertPreTokenizer` object does *not* deserialize as a `Whitespace`, which is only true because
/// the tag is required and distinct.
#[derive(Serialize, Deserialize)]
enum WhitespaceTag {
    Whitespace,
}

#[derive(Serialize, Deserialize)]
struct WhitespaceMirror {
    #[serde(rename = "type")]
    _type: WhitespaceTag,
}

pub mod whitespace {
    use super::*;

    pub fn serialize<S: Serializer>(_v: &Whitespace, s: S) -> Result<S::Ok, S::Error> {
        WhitespaceMirror {
            _type: WhitespaceTag::Whitespace,
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Whitespace, D::Error> {
        WhitespaceMirror::deserialize(d)?;
        Ok(Whitespace)
    }
}

// -------------------------------------------------------------------------------------------------
// WhitespaceSplit
// -------------------------------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
enum WhitespaceSplitTag {
    WhitespaceSplit,
}

#[derive(Serialize, Deserialize)]
struct WhitespaceSplitMirror {
    #[serde(rename = "type")]
    _type: WhitespaceSplitTag,
}

pub mod whitespace_split {
    use super::*;

    pub fn serialize<S: Serializer>(_v: &WhitespaceSplit, s: S) -> Result<S::Ok, S::Error> {
        WhitespaceSplitMirror {
            _type: WhitespaceSplitTag::WhitespaceSplit,
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<WhitespaceSplit, D::Error> {
        WhitespaceSplitMirror::deserialize(d)?;
        Ok(WhitespaceSplit)
    }
}

// -------------------------------------------------------------------------------------------------
// UnicodeScripts
// -------------------------------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
enum UnicodeScriptsTag {
    UnicodeScripts,
}

#[derive(Serialize, Deserialize)]
struct UnicodeScriptsMirror {
    #[serde(rename = "type")]
    _type: UnicodeScriptsTag,
}

pub mod unicode_scripts {
    use super::*;

    pub fn serialize<S: Serializer>(_v: &UnicodeScripts, s: S) -> Result<S::Ok, S::Error> {
        UnicodeScriptsMirror {
            _type: UnicodeScriptsTag::UnicodeScripts,
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<UnicodeScripts, D::Error> {
        UnicodeScriptsMirror::deserialize(d)?;
        Ok(UnicodeScripts::new())
    }
}

// -------------------------------------------------------------------------------------------------
// CharDelimiterSplit
// -------------------------------------------------------------------------------------------------

/// The tag is spelled `CharDelimiterSplit`, not `Delimiter` — that is what `impl_serde_type!` on a
/// struct of that name produced. `PreTokenizerWrapper::EnumType` has no such variant, so a config
/// written this way reaches the wrapper through the *untagged* legacy fallback. See the module docs.
#[derive(Serialize, Deserialize)]
enum CharDelimiterSplitTag {
    CharDelimiterSplit,
}

#[derive(Serialize, Deserialize)]
struct CharDelimiterSplitMirror {
    #[serde(rename = "type")]
    _type: CharDelimiterSplitTag,
    delimiter: char,
}

pub mod delimiter {
    use super::*;

    pub fn serialize<S: Serializer>(v: &CharDelimiterSplit, s: S) -> Result<S::Ok, S::Error> {
        CharDelimiterSplitMirror {
            _type: CharDelimiterSplitTag::CharDelimiterSplit,
            delimiter: v.delimiter,
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<CharDelimiterSplit, D::Error> {
        let m = CharDelimiterSplitMirror::deserialize(d)?;
        Ok(CharDelimiterSplit::new(m.delimiter))
    }
}

// -------------------------------------------------------------------------------------------------
// Digits
// -------------------------------------------------------------------------------------------------

/// `#[non_exhaustive]`, so it is built through `Digits::new` rather than by a `remote` derive's
/// struct literal.
#[derive(Serialize, Deserialize)]
enum DigitsTag {
    Digits,
}

#[derive(Serialize, Deserialize)]
struct DigitsMirror {
    #[serde(rename = "type")]
    _type: DigitsTag,
    individual_digits: bool,
}

pub mod digits {
    use super::*;

    pub fn serialize<S: Serializer>(v: &Digits, s: S) -> Result<S::Ok, S::Error> {
        DigitsMirror {
            _type: DigitsTag::Digits,
            individual_digits: v.individual_digits,
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Digits, D::Error> {
        let m = DigitsMirror::deserialize(d)?;
        Ok(Digits::new(m.individual_digits))
    }
}

// -------------------------------------------------------------------------------------------------
// FixedLength
// -------------------------------------------------------------------------------------------------

/// `length` keeps its `#[serde(default)]`, and the default is `5` — moved here from `tk-encode`,
/// where the function existed only to be named from a `#[serde(default = ...)]`.
fn default_length() -> usize {
    5
}

#[derive(Serialize, Deserialize)]
enum FixedLengthTag {
    FixedLength,
}

#[derive(Serialize, Deserialize)]
struct FixedLengthMirror {
    #[serde(rename = "type")]
    _type: FixedLengthTag,
    #[serde(default = "default_length")]
    length: usize,
}

pub mod fixed_length {
    use super::*;

    pub fn serialize<S: Serializer>(v: &FixedLength, s: S) -> Result<S::Ok, S::Error> {
        FixedLengthMirror {
            _type: FixedLengthTag::FixedLength,
            length: v.length,
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<FixedLength, D::Error> {
        let m = FixedLengthMirror::deserialize(d)?;
        Ok(FixedLength::new(m.length))
    }
}

// -------------------------------------------------------------------------------------------------
// Punctuation
// -------------------------------------------------------------------------------------------------

/// An absent `behavior` means `Isolated`, which is also `Punctuation::default()`. Moved here from
/// `tk-encode` for the same reason as [`default_length`].
fn default_split() -> SplitDelimiterBehavior {
    SplitDelimiterBehavior::Isolated
}

#[derive(Serialize, Deserialize)]
enum PunctuationTag {
    Punctuation,
}

#[derive(Serialize, Deserialize)]
struct PunctuationMirror {
    #[serde(rename = "type")]
    _type: PunctuationTag,
    #[serde(default = "default_split", with = "SplitDelimiterBehaviorDef")]
    behavior: SplitDelimiterBehavior,
}

pub mod punctuation {
    use super::*;

    pub fn serialize<S: Serializer>(v: &Punctuation, s: S) -> Result<S::Ok, S::Error> {
        PunctuationMirror {
            _type: PunctuationTag::Punctuation,
            behavior: v.behavior,
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Punctuation, D::Error> {
        let m = PunctuationMirror::deserialize(d)?;
        Ok(Punctuation::new(m.behavior))
    }
}

// -------------------------------------------------------------------------------------------------
// ByteLevel
// -------------------------------------------------------------------------------------------------

/// `use_regex` is the one pre-tokenizer field with a serde default, and it is `true`: configs
/// written before the field existed have to keep loading with the regex on.
fn default_true() -> bool {
    true
}

/// `#[non_exhaustive]`, so this is an explicit module rather than a `remote` derive — which is the
/// one place the pre-tokenizer `ByteLevel` differs from `ByteLevelDecoder`, whose three identically
/// named fields serde *can* drive directly.
///
/// This mirror is also what `PostProcessorWrapper::ByteLevel` names: the post-processor and the
/// pre-tokenizer are the same type (`tk_encode::processors` re-exports `pre_tokenizers::byte_level`),
/// so there is one shape and one mirror of it, on the side that owns the definition.
#[derive(Serialize, Deserialize)]
enum ByteLevelTag {
    ByteLevel,
}

#[derive(Serialize, Deserialize)]
struct ByteLevelMirror {
    #[serde(rename = "type")]
    _type: ByteLevelTag,
    add_prefix_space: bool,
    trim_offsets: bool,
    #[serde(default = "default_true")]
    use_regex: bool,
}

pub mod byte_level {
    use super::*;

    pub fn serialize<S: Serializer>(v: &ByteLevel, s: S) -> Result<S::Ok, S::Error> {
        ByteLevelMirror {
            _type: ByteLevelTag::ByteLevel,
            add_prefix_space: v.add_prefix_space,
            trim_offsets: v.trim_offsets,
            use_regex: v.use_regex,
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<ByteLevel, D::Error> {
        let m = ByteLevelMirror::deserialize(d)?;
        Ok(ByteLevel::new(
            m.add_prefix_space,
            m.trim_offsets,
            m.use_regex,
        ))
    }
}

// -------------------------------------------------------------------------------------------------
// Metaspace
// -------------------------------------------------------------------------------------------------

/// `replacement` is private behind a getter (it keeps the derived `str_rep` in sync), so this one
/// converts explicitly rather than using `remote`.
///
/// The legacy rule, reproduced exactly: an absent `prepend_scheme` means `Always`, **not** `Never`;
/// `add_prefix_space: false` is accepted only alongside an explicit `prepend_scheme: "never"` and is
/// otherwise a hard error. `str_rep` is read and thrown away. This is the same rule, word for word,
/// as `decoders::mirror::metaspace`'s — the pre-tokenizer and the decoder have to agree about what a
/// `tokenizer.json` means, and both are load-bearing for ids, so neither gets "fixed".
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
    #[serde(default = "default_prepend_scheme", with = "PrependSchemeDef")]
    prepend_scheme: PrependScheme,
    split: Option<bool>,
    #[serde(rename = "str_rep")]
    _str_rep: Option<String>,
}

fn default_prepend_scheme() -> PrependScheme {
    PrependScheme::Always
}

/// `str_rep` is not written out: it was `#[serde(skip)]` on the struct, being derived from
/// `replacement`. So the shape is `type`, `replacement`, `prepend_scheme`, `split` — the struct's own
/// declaration order, minus the skipped field.
#[derive(Serialize)]
#[serde(tag = "type", rename = "Metaspace")]
struct MetaspaceOut<'a> {
    replacement: char,
    #[serde(with = "PrependSchemeDef")]
    prepend_scheme: &'a PrependScheme,
    split: bool,
}

pub mod metaspace {
    use super::*;

    pub fn serialize<S: Serializer>(v: &Metaspace, s: S) -> Result<S::Ok, S::Error> {
        MetaspaceOut {
            replacement: v.get_replacement(),
            prepend_scheme: &v.prepend_scheme,
            split: v.split,
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Metaspace, D::Error> {
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

/// `Split` keeps a compiled matcher and possibly a native FSM, both derived from `pattern`, so it
/// can only be built through its constructor — which can fail, on a pattern the regex backend
/// rejects and the FSM does not recognise. `search` and `fsm` were `#[serde(skip)]` for the same
/// reason and stay off the wire.
#[derive(Deserialize)]
enum SplitTag {
    Split,
}

#[derive(Deserialize)]
struct SplitIn {
    #[serde(rename = "type")]
    _type: SplitTag,
    #[serde(with = "SplitPatternDef")]
    pattern: SplitPattern,
    #[serde(with = "SplitDelimiterBehaviorDef")]
    behavior: SplitDelimiterBehavior,
    invert: bool,
}

#[derive(Serialize)]
#[serde(tag = "type", rename = "Split")]
struct SplitOut<'a> {
    #[serde(with = "SplitPatternDef")]
    pattern: &'a SplitPattern,
    #[serde(with = "SplitDelimiterBehaviorDef")]
    behavior: &'a SplitDelimiterBehavior,
    invert: bool,
}

pub mod split {
    use super::*;

    pub fn serialize<S: Serializer>(v: &Split, s: S) -> Result<S::Ok, S::Error> {
        SplitOut {
            pattern: &v.pattern,
            behavior: &v.behavior,
            invert: v.invert,
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Split, D::Error> {
        let helper = SplitIn::deserialize(d)?;
        Split::new(helper.pattern, helper.behavior, helper.invert).map_err(D::Error::custom)
    }
}

// -------------------------------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------------------------------

/// Every test here moved out of `tk-encode` with the serde it exercises.
///
/// They are deliberately written against the *leaf* mirrors rather than against
/// [`super::PreTokenizerWrapper`], because that is what they were checking: several assert that one
/// pre-tokenizer refuses a tag belonging to another, and the wrapper would hide that by simply
/// picking the other variant. `#[serde(with = ...)]` needs a literal path, so each one gets a small
/// newtype rather than a generic helper.
#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Deserialize)]
    struct PunctuationJson(#[serde(with = "punctuation")] Punctuation);

    #[derive(Deserialize)]
    struct ByteLevelJson(#[serde(with = "byte_level")] ByteLevel);

    #[derive(Serialize, Deserialize)]
    struct MetaspaceJson(#[serde(with = "metaspace")] Metaspace);

    #[derive(Serialize, Deserialize)]
    struct SplitJson(#[serde(with = "split")] Split);

    #[derive(Serialize)]
    struct PrependSchemeJson<'a>(#[serde(with = "PrependSchemeDef")] &'a PrependScheme);

    /// `tk-encode` spells `PrependScheme`'s `Display` out by hand so the name survives a build with
    /// no serde in it. This is the test that stops the two from drifting; it lives here now because
    /// the `rename_all = "snake_case"` half is here.
    #[test]
    fn display_matches_serde() {
        for scheme in [
            PrependScheme::First,
            PrependScheme::Never,
            PrependScheme::Always,
        ] {
            let via_serde = serde_json::to_string(&PrependSchemeJson(&scheme)).unwrap();
            assert_eq!(format!("\"{scheme}\""), via_serde);
        }
    }

    /// An absent `behavior` defaults to `Isolated`, which is `Punctuation::default()`.
    #[test]
    fn punctuation_deserialization() {
        let punctuation = serde_json::from_str::<PunctuationJson>(r#"{"type": "Punctuation"}"#)
            .unwrap()
            .0;
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
            serde_json::from_str::<PunctuationJson>(r#"{"type": "WhitespaceSplit"}"#).unwrap();
    }

    /// `use_regex` was added after `ByteLevel` shipped, so an object without it has to keep loading
    /// with the regex on.
    #[test]
    fn byte_level_deserialization() {
        // Before use_regex
        let byte_level = serde_json::from_str::<ByteLevelJson>(
            r#"{"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false}"#,
        )
        .unwrap()
        .0;
        assert!(byte_level.use_regex);

        // Loading works, new future BC test.
        let byte_level = serde_json::from_str::<ByteLevelJson>(
            r#"{"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": true}"#,
        )
        .unwrap()
        .0;
        assert!(byte_level.use_regex);

        let byte_level = serde_json::from_str::<ByteLevelJson>(
            r#"{"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": false}"#,
        )
        .unwrap()
        .0;
        assert!(!byte_level.use_regex);
    }

    /// The legacy `add_prefix_space` / `str_rep` rule, exactly as it was asserted in `tk-encode`.
    #[test]
    fn metaspace_serialization() {
        let metaspace = Metaspace::new('_', PrependScheme::Always, true);
        let metaspace_s =
            r#"{"type":"Metaspace","replacement":"_","prepend_scheme":"always","split":true}"#;
        assert_eq!(
            serde_json::to_string(&MetaspaceJson(metaspace.clone())).unwrap(),
            metaspace_s
        );
        assert_eq!(
            serde_json::from_str::<MetaspaceJson>(metaspace_s)
                .unwrap()
                .0,
            metaspace
        );

        // Also check it can deserialize previous versions
        let metaspace_s = r#"{"type":"Metaspace","replacement":"_","add_prefix_space":false,"prepend_scheme":"always"}"#;
        assert!(serde_json::from_str::<MetaspaceJson>(metaspace_s).is_err(),);

        let metaspace = Metaspace::new('_', PrependScheme::Always, true);
        let metaspace_s = r#"{"type":"Metaspace","str_rep":"_","replacement":"_","add_prefix_space":true,"prepend_scheme":"always"}"#;
        assert_eq!(
            serde_json::from_str::<MetaspaceJson>(metaspace_s)
                .unwrap()
                .0,
            metaspace
        );

        let metaspace_parsed = serde_json::from_str::<MetaspaceJson>(
            r#"{"type":"Metaspace","replacement":"_","add_prefix_space":true}"#,
        )
        .unwrap()
        .0;
        assert_eq!(metaspace_parsed, metaspace);
    }

    /// A config spelling its pattern as a string must also *deserialize* with no backend — the regex
    /// half of `split_serialization` below can only run once one is compiled.
    #[test]
    fn a_string_pattern_deserializes_with_no_backend() {
        let split_s =
            r#"{"type":"Split","pattern":{"String":"Hello"},"behavior":"Removed","invert":true}"#;
        let split = Split::new("Hello", SplitDelimiterBehavior::Removed, true).unwrap();
        assert_eq!(serde_json::from_str::<SplitJson>(split_s).unwrap().0, split);
        assert_eq!(
            serde_json::to_string(&SplitJson(split.clone())).unwrap(),
            split_s
        );
    }

    #[cfg(feature = "fancy-regex")] // needs a system-regex backend
    #[test]
    fn split_serialization() {
        use SplitDelimiterBehavior::*;

        let split = Split::new("Hello", Removed, true).unwrap();
        let split_s =
            r#"{"type":"Split","pattern":{"String":"Hello"},"behavior":"Removed","invert":true}"#;
        assert_eq!(
            serde_json::to_string(&SplitJson(split.clone())).unwrap(),
            split_s
        );
        assert_eq!(serde_json::from_str::<SplitJson>(split_s).unwrap().0, split);

        let split = Split::new(SplitPattern::Regex(r"\s+".into()), Isolated, false).unwrap();
        let split_s =
            r#"{"type":"Split","pattern":{"Regex":"\\s+"},"behavior":"Isolated","invert":false}"#;
        assert_eq!(
            serde_json::to_string(&SplitJson(split.clone())).unwrap(),
            split_s
        );
        assert_eq!(serde_json::from_str::<SplitJson>(split_s).unwrap().0, split);
    }
}
