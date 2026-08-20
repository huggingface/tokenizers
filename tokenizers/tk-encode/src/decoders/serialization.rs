//! serde for every decoder: `tk-encode` defines the ten decoder types and serializes none of them.
//!
//! Rust's orphan rule is the whole reason this file exists: `Serialize` and `Deserialize` are
//! foreign traits and `ByteLevelDecoder`/`MetaspaceDecoder`/`ReplaceDecoder` and the rest are
//! foreign types, so this crate cannot implement one for the other. What it *can* do is define a
//! local mirror of the JSON shape and convert — which is what `#[serde(with = "...")]` on the
//! wrapper's variants asks for.
//!
//! The mirrors are the authority on the on-disk shape, including the backwards-compatible spellings
//! (`Metaspace`'s legacy `add_prefix_space` and its dead `str_rep`). That is the point of moving
//! them here: the runtime crate no longer knows or cares how a decoder is written down.
//!
//! ## Two mirror shapes, and how to pick
//!
//! * **`remote`** — serde drives the foreign type directly. Only usable when every field is `pub`
//!   *and* the type is not `#[non_exhaustive]`: the generated code builds the type with a struct
//!   literal, which `#[non_exhaustive]` forbids across a crate boundary. `ByteLevelDecoder` is the
//!   one decoder that qualifies.
//! * **an explicit `mod` with `serialize`/`deserialize`** — a local struct carrying *the same serde
//!   attributes the type used to carry*, converted through the type's own constructor. Everything
//!   else uses this, either because a field is private behind a getter (`Metaspace`, `Replace`) or
//!   because the type is `#[non_exhaustive]` (`BPEDecoder`, `CTC`, `Strip`, `WordPiece`, `Fuse`,
//!   `ByteFallback`).
//!
//! ## Whether the `"type"` tag is required is per-decoder, and it is load-bearing
//!
//! `DecoderWrapper`'s legacy fallback is an *untagged* enum, so a variant that is lenient about the
//! tag will happily claim a tag-less object that should have been rejected. As `crate::macros`
//! spells out, a bare `#[serde(tag = "type")]` on a struct does **not** make the tag required — it
//! only adds it on the way out. So the tag is:
//!
//! * **optional** for `BPEDecoder`, `CTC`, `Strip` and `WordPiece`, which carried a bare
//!   `#[serde(tag = "type")]`. Their mirrors carry the identical attribute, so the codegen — and
//!   therefore the accepted input — is unchanged. They are still rejected for a tag-less object in
//!   practice, but by a *missing required field*, not by the tag.
//! * **required** for `Fuse` and `ByteFallback`, which had no fields at all and so used a
//!   `monostate::MustBe!` type tag as their only required field. Nothing else would have stopped
//!   `{}` from deserializing as a `Fuse`, and `decoder_serialization_no_decode` is the test that
//!   catches it. Their mirrors reproduce the requirement with a one-variant tag enum, which is also
//!   what let `monostate` be dropped from `tk-encode` entirely.

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use tk_encode::decoders::bpe::BPEDecoder;
use tk_encode::decoders::byte_fallback::ByteFallback;
use tk_encode::decoders::byte_level::ByteLevelDecoder;
use tk_encode::decoders::ctc::CTC;
use tk_encode::decoders::fuse::Fuse;
use tk_encode::decoders::metaspace::{MetaspaceDecoder, PrependScheme};
use tk_encode::decoders::replace::ReplaceDecoder;
use tk_encode::decoders::strip::Strip;
use tk_encode::decoders::wordpiece::WordPiece;
use tk_encode::utils::search::ReplacePattern;

fn default_true() -> bool {
    true
}

/// `ReplacePattern` is a plain two-variant enum with no invariants, so `remote` drives it directly.
/// Mirrored here rather than derived in `tk-encode` for the same reason as everything else in this
/// file: the runtime crate links no serde.
#[derive(Serialize, Deserialize)]
#[serde(remote = "ReplacePattern")]
pub enum ReplacePatternDef {
    String(String),
    Regex(String),
}

// ---------------------------------------------------------------------------------------------
// ByteLevel
// ---------------------------------------------------------------------------------------------

/// All three fields are public and there is no invariant between them, so serde's `remote` derive
/// can drive the foreign type directly and no hand-written conversion is needed.
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename = "ByteLevel", remote = "ByteLevelDecoder")]
pub struct ByteLevelDecoderDef {
    pub add_prefix_space: bool,
    pub trim_offsets: bool,
    // The one decoder field with a serde default, and it is `true`.
    #[serde(default = "default_true")]
    pub use_regex: bool,
}

// ---------------------------------------------------------------------------------------------
// Metaspace
// ---------------------------------------------------------------------------------------------

/// `replacement` is private behind a getter (it used to keep a derived `str_rep` in sync), so this
/// one converts explicitly rather than using `remote`.
///
/// The legacy rule, reproduced exactly: an absent `prepend_scheme` means `Always`, **not** `Never`;
/// `add_prefix_space: false` is accepted only alongside an explicit `prepend_scheme: "never"` and
/// is otherwise a hard error. `str_rep` is read and thrown away.
/// The tag is a mandatory field rather than `#[serde(tag = ...)]`, which is how the original
/// `MetaspaceHelper` spelled it. It matters: `DecoderWrapper`'s legacy fallback is an *untagged*
/// enum, so a lenient variant will happily claim a tag-less object that should have been rejected
/// -- which is exactly what `decoder_serialization_no_decode` catches.
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
    // `PrependScheme` is `pre_tokenizers::metaspace`'s type, re-exported by `decoders::metaspace`.
    // One type, one mirror: `pre_tokenizers::mirror::PrependSchemeDef`, on the side that defines it.
    #[serde(
        default = "default_prepend_scheme",
        with = "crate::pre_tokenizers::mirror::PrependSchemeDef"
    )]
    prepend_scheme: PrependScheme,
    split: Option<bool>,
    #[serde(rename = "str_rep")]
    _str_rep: Option<String>,
}

fn default_prepend_scheme() -> PrependScheme {
    PrependScheme::Always
}

#[derive(Serialize)]
#[serde(tag = "type", rename = "Metaspace")]
struct MetaspaceOut<'a> {
    replacement: char,
    #[serde(with = "crate::pre_tokenizers::mirror::PrependSchemeDef")]
    prepend_scheme: &'a PrependScheme,
    split: bool,
}

pub mod metaspace {
    use super::*;

    pub fn serialize<S: Serializer>(v: &MetaspaceDecoder, s: S) -> Result<S::Ok, S::Error> {
        MetaspaceOut {
            replacement: v.get_replacement(),
            prepend_scheme: &v.prepend_scheme,
            split: v.split,
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<MetaspaceDecoder, D::Error> {
        let mut helper = MetaspaceIn::deserialize(d)?;
        if helper.add_prefix_space == Some(false) {
            if helper.prepend_scheme != PrependScheme::Never {
                return Err(D::Error::custom(
                    "add_prefix_space does not match declared prepend_scheme",
                ));
            }
            helper.prepend_scheme = PrependScheme::Never;
        }
        Ok(MetaspaceDecoder::new(
            helper.replacement,
            helper.prepend_scheme,
            helper.split.unwrap_or(true),
        ))
    }
}

// ---------------------------------------------------------------------------------------------
// Replace
// ---------------------------------------------------------------------------------------------

/// `ReplaceDecoder` keeps a compiled matcher derived from `pattern`, so it can only be built
/// through its constructor -- which can fail, on a pattern the regex backend rejects.
#[derive(Deserialize)]
enum ReplaceTag {
    Replace,
}

#[derive(Deserialize)]
struct ReplaceIn {
    #[serde(rename = "type")]
    _type: ReplaceTag,
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

    pub fn serialize<S: Serializer>(v: &ReplaceDecoder, s: S) -> Result<S::Ok, S::Error> {
        ReplaceOut {
            pattern: v.pattern(),
            content: v.content(),
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<ReplaceDecoder, D::Error> {
        let helper = ReplaceIn::deserialize(d)?;
        ReplaceDecoder::new(helper.pattern, helper.content).map_err(D::Error::custom)
    }
}

// ---------------------------------------------------------------------------------------------
// BPEDecoder
// ---------------------------------------------------------------------------------------------

/// The tag is spelled `BPEDecoder`, not `BPE` — that is what `#[serde(tag = "type")]` on a struct
/// called `BPEDecoder` used to produce, and `DecoderWrapper`'s `EnumType` names the same spelling.
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename = "BPEDecoder")]
struct BPEDecoderMirror {
    suffix: String,
}

pub mod bpe {
    use super::*;

    pub fn serialize<S: Serializer>(v: &BPEDecoder, s: S) -> Result<S::Ok, S::Error> {
        BPEDecoderMirror {
            suffix: v.suffix.clone(),
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<BPEDecoder, D::Error> {
        let m = BPEDecoderMirror::deserialize(d)?;
        Ok(BPEDecoder::new(m.suffix))
    }
}

// ---------------------------------------------------------------------------------------------
// ByteFallback
// ---------------------------------------------------------------------------------------------

/// No fields, so the tag is the only thing on the wire — and the only thing that can make `{}` fail.
/// See the module docs: this is one of the two decoders where the tag is *required*.
#[derive(Serialize, Deserialize)]
enum ByteFallbackTag {
    ByteFallback,
}

#[derive(Serialize, Deserialize)]
struct ByteFallbackMirror {
    #[serde(rename = "type")]
    _type: ByteFallbackTag,
}

pub mod byte_fallback {
    use super::*;

    pub fn serialize<S: Serializer>(_v: &ByteFallback, s: S) -> Result<S::Ok, S::Error> {
        ByteFallbackMirror {
            _type: ByteFallbackTag::ByteFallback,
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<ByteFallback, D::Error> {
        ByteFallbackMirror::deserialize(d)?;
        Ok(ByteFallback::new())
    }
}

// ---------------------------------------------------------------------------------------------
// CTC
// ---------------------------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename = "CTC")]
struct CTCMirror {
    pad_token: String,
    word_delimiter_token: String,
    cleanup: bool,
}

pub mod ctc {
    use super::*;

    pub fn serialize<S: Serializer>(v: &CTC, s: S) -> Result<S::Ok, S::Error> {
        CTCMirror {
            pad_token: v.pad_token.clone(),
            word_delimiter_token: v.word_delimiter_token.clone(),
            cleanup: v.cleanup,
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<CTC, D::Error> {
        let m = CTCMirror::deserialize(d)?;
        Ok(CTC::new(m.pad_token, m.word_delimiter_token, m.cleanup))
    }
}

// ---------------------------------------------------------------------------------------------
// Fuse
// ---------------------------------------------------------------------------------------------

/// The other tag-required decoder; see `ByteFallback` above and the module docs.
#[derive(Serialize, Deserialize)]
enum FuseTag {
    Fuse,
}

#[derive(Serialize, Deserialize)]
struct FuseMirror {
    #[serde(rename = "type")]
    _type: FuseTag,
}

pub mod fuse {
    use super::*;

    pub fn serialize<S: Serializer>(_v: &Fuse, s: S) -> Result<S::Ok, S::Error> {
        FuseMirror {
            _type: FuseTag::Fuse,
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Fuse, D::Error> {
        FuseMirror::deserialize(d)?;
        Ok(Fuse::new())
    }
}

// ---------------------------------------------------------------------------------------------
// Strip
// ---------------------------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename = "Strip")]
struct StripMirror {
    content: char,
    start: usize,
    stop: usize,
}

pub mod strip {
    use super::*;

    pub fn serialize<S: Serializer>(v: &Strip, s: S) -> Result<S::Ok, S::Error> {
        StripMirror {
            content: v.content,
            start: v.start,
            stop: v.stop,
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Strip, D::Error> {
        let m = StripMirror::deserialize(d)?;
        Ok(Strip::new(m.content, m.start, m.stop))
    }
}

// ---------------------------------------------------------------------------------------------
// WordPiece
// ---------------------------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename = "WordPiece")]
struct WordPieceMirror {
    prefix: String,
    cleanup: bool,
}

pub mod wordpiece {
    use super::*;

    pub fn serialize<S: Serializer>(v: &WordPiece, s: S) -> Result<S::Ok, S::Error> {
        WordPieceMirror {
            prefix: v.prefix.clone(),
            cleanup: v.cleanup,
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<WordPiece, D::Error> {
        let m = WordPieceMirror::deserialize(d)?;
        Ok(WordPiece::new(m.prefix, m.cleanup))
    }
}
