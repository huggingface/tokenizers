//! The decoders whose serde cannot be a derive on the type.
//!
//! Five of the ten decoders are plain data and carry `#[cfg_attr(feature = "serde", derive(…))]` in
//! their own file, next to the struct. The five here need a hand-written impl:
//!
//! * `MetaspaceDecoder` and `ReplaceDecoder`, because a field is private behind a getter and the
//!   value can only be rebuilt through a constructor — `ReplaceDecoder`'s can fail, on a pattern the
//!   regex backend rejects;
//! * `Fuse`, `ByteFallback` and `ByteLevelDecoder`, because they have no fields at all, and
//!   something has to make the `"type"` tag *required*.
//!
//! ## Whether the `"type"` tag is required is per-decoder, and it is load-bearing
//!
//! `DecoderWrapper`'s legacy fallback is an *untagged* enum, so a variant that is lenient about the
//! tag will happily claim a tag-less object that should have been rejected. A bare
//! `#[serde(tag = "type")]` on a struct does **not** make the tag required — it only adds it on the
//! way out. So the tag is:
//!
//! * **optional** for `BPEDecoder`, `CTC`, `Strip` and `WordPiece`, which carry that bare attribute.
//!   They are still rejected for a tag-less object in practice, but by a *missing required field*,
//!   not by the tag.
//! * **required** for `Metaspace`, `Replace`, `Fuse`, `ByteFallback` and `ByteLevelDecoder`,
//!   spelled here as a one-variant tag enum in a mandatory field. For the field-less ones nothing
//!   else would stop
//!   `{}` from deserializing as a `Fuse`, and `decoder_serialization_no_decode` is the test that
//!   catches it. That trick is also what keeps `monostate` out of this crate: the field-less
//!   decoders used to carry a `monostate::MustBe!("Fuse")` for exactly this job.
//!
//! A required tag as a mandatory *field* on the way in, written first on the way out, is
//! byte-identical to what `#[serde(tag = "type")]` emits: `"type"` first, then declaration order.

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::byte_fallback::ByteFallback;
use super::byte_level::ByteLevelDecoder;
use super::fuse::Fuse;
use super::metaspace::{MetaspaceDecoder, PrependScheme};
use super::replace::ReplaceDecoder;
use crate::utils::search::ReplacePattern;

// ---------------------------------------------------------------------------------------------
// Metaspace
// ---------------------------------------------------------------------------------------------

/// The legacy rule, reproduced exactly: an absent `prepend_scheme` means `Always`, **not** `Never`;
/// `add_prefix_space: false` is accepted only alongside an explicit `prepend_scheme: "never"` and is
/// otherwise a hard error. `str_rep` is read and thrown away.
///
/// This is the same rule, word for word, as the `Metaspace` *pre-tokenizer*'s — the two have to
/// agree about what a `tokenizer.json` means, and both are load-bearing for ids, so neither gets
/// "fixed".
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

/// `str_rep` is derived from `replacement` and never written; the shape is `type`, `replacement`,
/// `prepend_scheme`, `split`.
#[derive(Serialize)]
#[serde(tag = "type", rename = "Metaspace")]
struct MetaspaceOut<'a> {
    replacement: char,
    prepend_scheme: &'a PrependScheme,
    split: bool,
}

impl Serialize for MetaspaceDecoder {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        MetaspaceOut {
            replacement: self.get_replacement(),
            prepend_scheme: &self.prepend_scheme,
            split: self.split,
        }
        .serialize(s)
    }
}

impl<'de> Deserialize<'de> for MetaspaceDecoder {
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

/// `ReplaceDecoder` keeps a compiled matcher derived from `pattern`, so it can only be built through
/// its constructor -- which can fail, on a pattern the regex backend rejects. The `search` field
/// never reaches the wire.
#[derive(Deserialize)]
enum ReplaceTag {
    Replace,
}

#[derive(Deserialize)]
struct ReplaceIn {
    #[serde(rename = "type")]
    _type: ReplaceTag,
    pattern: ReplacePattern,
    content: String,
}

#[derive(Serialize)]
#[serde(tag = "type", rename = "Replace")]
struct ReplaceOut<'a> {
    pattern: &'a ReplacePattern,
    content: &'a str,
}

impl Serialize for ReplaceDecoder {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        ReplaceOut {
            pattern: self.pattern(),
            content: self.content(),
        }
        .serialize(s)
    }
}

impl<'de> Deserialize<'de> for ReplaceDecoder {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let helper = ReplaceIn::deserialize(d)?;
        ReplaceDecoder::new(helper.pattern, helper.content).map_err(D::Error::custom)
    }
}

// ---------------------------------------------------------------------------------------------
// ByteFallback, Fuse and ByteLevel — no fields, so the tag is the only thing on the wire
// ---------------------------------------------------------------------------------------------

/// One `Serialize`/`Deserialize` pair for a field-less decoder, where the required tag is all there
/// is. The tag is spelled out rather than taken from the type name, because the two differ for
/// `ByteLevelDecoder`, whose tag on disk is `ByteLevel`.
macro_rules! tag_only {
    ($ty:ident, $tag:ident, $ctor:expr) => {
        paste::paste! {
            #[derive(Serialize, Deserialize)]
            enum [<$ty Tag>] {
                $tag,
            }

            #[derive(Serialize, Deserialize)]
            struct [<$ty Helper>] {
                #[serde(rename = "type")]
                _type: [<$ty Tag>],
            }

            impl Serialize for $ty {
                fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                    [<$ty Helper>] { _type: [<$ty Tag>]::$tag }.serialize(s)
                }
            }

            impl<'de> Deserialize<'de> for $ty {
                fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                    [<$ty Helper>]::deserialize(d)?;
                    Ok($ctor)
                }
            }
        }
    };
}

tag_only!(Fuse, Fuse, Fuse::new());
tag_only!(ByteFallback, ByteFallback, ByteFallback::new());
tag_only!(ByteLevelDecoder, ByteLevel, ByteLevelDecoder::new());
