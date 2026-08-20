//! serde for the *shared runtime types* — the ones that are not components of a pipeline but still
//! appear in a `tokenizer.json` or in a pickled binding object.
//!
//! `tk-encode`'s per-component mirrors live next to the wrapper they belong to
//! (`decoders::mirror`, `normalizers::mirror`, …). What is left over are the types that cut across
//! all of them: the padding and truncation parameters that hang off a `Tokenizer` rather than off
//! any one component, the `SplitDelimiterBehavior` that half the pre-tokenizers and normalizers
//! carry as a field, `AddedToken`, `Encoding`, and the `ProgressFormat` that a trainer records.
//!
//! Same orphan-rule reason as everywhere else in this crate: those types are defined in
//! `tk-encode`, `Serialize`/`Deserialize` are defined in `serde`, and this crate can implement
//! neither for the other. It defines a local mirror and converts.
//!
//! Most of these are plain data with every field `pub` and no `#[non_exhaustive]`, which is exactly
//! the case serde's `remote` derive handles: it drives the foreign type directly and no
//! hand-written conversion is needed. `Encoding` is the one exception, because its nine fields are
//! private — it gets an explicit pair of borrow/own mirrors instead.

use std::collections::BTreeMap;
use std::ops::Range;

use ahash::AHashMap;
use serde::ser::SerializeSeq;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use tk_encode::Encoding;
use tk_encode::tokenizer::Offsets;
use tk_encode::tokenizer::normalizer::SplitDelimiterBehavior;
use tk_encode::utils::padding::{PaddingDirection, PaddingParams, PaddingStrategy};
use tk_encode::utils::progress::ProgressFormat;
use tk_encode::utils::truncation::{TruncationDirection, TruncationParams, TruncationStrategy};
use tk_encode::vocab::bucket_added_vocabulary::{AddedToken, AddedVocabulary};

// -------------------------------------------------------------------------------------------------
// ordered_map
// -------------------------------------------------------------------------------------------------

/// Serialize an [`AHashMap`] through a [`BTreeMap`] so the output has a deterministic key order.
///
/// Only ever named from a `#[serde(serialize_with = ...)]` — today just the `TemplateProcessing`
/// special-token table — so it moved here with the serde it exists to serve. A hash map's iteration
/// order is unspecified, and a `tokenizer.json` that reorders its keys between two saves of the
/// same tokenizer is a diff nobody wants to read.
pub fn ordered_map<S, K, V>(value: &AHashMap<K, V>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
    K: Serialize + std::cmp::Ord,
    V: Serialize,
{
    let ordered: BTreeMap<_, _> = value.iter().collect();
    ordered.serialize(serializer)
}

// -------------------------------------------------------------------------------------------------
// SplitDelimiterBehavior
// -------------------------------------------------------------------------------------------------

/// Carried as a field by `Split`, `Punctuation`, `Digits`, `UnicodeScripts` and the whitespace
/// pre-tokenizers, so every one of those mirrors names this one with `#[serde(with = ...)]`.
///
/// `tk-encode` spells out its `Display` by hand and has a `display_matches_serde` test to keep the
/// two from drifting. That test asserts a property of *this* mirror, so it moved here with it.
#[derive(Serialize, Deserialize)]
#[serde(remote = "SplitDelimiterBehavior")]
pub enum SplitDelimiterBehaviorDef {
    Removed,
    Isolated,
    MergedWithPrevious,
    MergedWithNext,
    Contiguous,
}

// -------------------------------------------------------------------------------------------------
// Truncation
// -------------------------------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
#[serde(remote = "TruncationDirection")]
pub enum TruncationDirectionDef {
    Left,
    Right,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "TruncationStrategy")]
pub enum TruncationStrategyDef {
    LongestFirst,
    OnlyFirst,
    OnlySecond,
}

/// `direction` keeps its `#[serde(default)]`: a `truncation` block written before the field existed
/// has to keep loading, and `test_deserialize_defaults` is the test that says so.
///
/// A standalone mirror with `From` conversions rather than a `remote` derive, because the caller
/// needs it inside an `Option`: `TokenizerImpl`'s field is `Option<TruncationParams>`, and
/// `#[serde(with = ...)]` does not reach through an `Option`. `Option<TruncationParamsMirror>` does,
/// for free.
#[derive(Serialize, Deserialize)]
pub struct TruncationParamsMirror {
    #[serde(default, with = "TruncationDirectionDef")]
    pub direction: TruncationDirection,
    pub max_length: usize,
    #[serde(with = "TruncationStrategyDef")]
    pub strategy: TruncationStrategy,
    pub stride: usize,
}

impl From<&TruncationParams> for TruncationParamsMirror {
    fn from(p: &TruncationParams) -> Self {
        Self {
            direction: p.direction,
            max_length: p.max_length,
            strategy: p.strategy,
            stride: p.stride,
        }
    }
}

impl From<TruncationParamsMirror> for TruncationParams {
    fn from(m: TruncationParamsMirror) -> Self {
        Self {
            direction: m.direction,
            max_length: m.max_length,
            strategy: m.strategy,
            stride: m.stride,
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Padding
// -------------------------------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
#[serde(remote = "PaddingDirection")]
pub enum PaddingDirectionDef {
    Left,
    Right,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "PaddingStrategy")]
pub enum PaddingStrategyDef {
    BatchLongest,
    Fixed(usize),
}

/// Same `Option`-shaped reason as [`TruncationParamsMirror`] for being a standalone mirror rather
/// than a `remote` derive.
#[derive(Serialize, Deserialize)]
pub struct PaddingParamsMirror {
    #[serde(with = "PaddingStrategyDef")]
    pub strategy: PaddingStrategy,
    #[serde(with = "PaddingDirectionDef")]
    pub direction: PaddingDirection,
    pub pad_to_multiple_of: Option<usize>,
    pub pad_id: u32,
    pub pad_type_id: u32,
    pub pad_token: String,
}

impl From<&PaddingParams> for PaddingParamsMirror {
    fn from(p: &PaddingParams) -> Self {
        Self {
            strategy: p.strategy.clone(),
            direction: p.direction,
            pad_to_multiple_of: p.pad_to_multiple_of,
            pad_id: p.pad_id,
            pad_type_id: p.pad_type_id,
            pad_token: p.pad_token.clone(),
        }
    }
}

impl From<PaddingParamsMirror> for PaddingParams {
    fn from(m: PaddingParamsMirror) -> Self {
        Self {
            strategy: m.strategy,
            direction: m.direction,
            pad_to_multiple_of: m.pad_to_multiple_of,
            pad_id: m.pad_id,
            pad_type_id: m.pad_type_id,
            pad_token: m.pad_token,
        }
    }
}

// -------------------------------------------------------------------------------------------------
// ProgressFormat
// -------------------------------------------------------------------------------------------------

/// A *training* setting, not a config one, but the trainers in `tk-train` derive serde on the
/// structs that carry it, so it needs a mirror all the same.
#[derive(Serialize, Deserialize)]
#[serde(remote = "ProgressFormat")]
pub enum ProgressFormatDef {
    Indicatif,
    JsonLines,
    Silent,
}

// -------------------------------------------------------------------------------------------------
// AddedToken
// -------------------------------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
#[serde(remote = "AddedToken")]
pub struct AddedTokenDef {
    pub content: String,
    pub single_word: bool,
    pub lstrip: bool,
    pub rstrip: bool,
    pub normalized: bool,
    pub special: bool,
}

/// One entry of the `added_tokens` array: an [`AddedToken`] with the id it was assigned, flattened
/// into a single object.
#[derive(Serialize, Deserialize)]
pub struct AddedTokenWithId {
    /// The id assigned to this token
    pub id: u32,
    /// The target AddedToken
    #[serde(flatten, with = "AddedTokenDef")]
    pub token: AddedToken,
}

/// Serialize an [`AddedVocabulary`] as its `added_tokens` array, ordered by id.
///
/// What goes out is the *logical* token list — id, content and flags — and never the derived
/// `Buckets`/`VocabStore`, which are rebuilt from this list by `add_tokens` on the way back in.
///
/// Note this is the *runtime* `AddedVocabulary` from `tk-encode`. The legacy config one is this
/// crate's own `tokenizer::added_vocabulary::AddedVocabulary`, which keeps its own derive: it is a
/// local type, so the orphan rule never applied to it.
pub mod added_vocabulary {
    use super::*;

    pub fn serialize<S: Serializer>(v: &AddedVocabulary, s: S) -> Result<S::Ok, S::Error> {
        let added_tokens: Vec<AddedTokenWithId> = v
            .get_added_tokens_decoder()
            .into_iter()
            .map(|(id, token)| AddedTokenWithId { id, token })
            .collect();
        let mut order: Vec<u64> = added_tokens
            .iter()
            .enumerate()
            .map(|(i, t)| ((t.id as u64) << 32) | i as u64)
            .collect();
        order.sort_unstable();

        let mut seq = s.serialize_seq(Some(added_tokens.len()))?;
        for key in &order {
            seq.serialize_element(&added_tokens[(*key & 0xFFFF_FFFF) as usize])?;
        }
        seq.end()
    }
}

// -------------------------------------------------------------------------------------------------
// Encoding
// -------------------------------------------------------------------------------------------------

/// [`Encoding`] is the one shared type whose fields are all private, so `remote` cannot drive it:
/// the generated code would need a struct literal. It gets a borrowing mirror for the way out and
/// an owning one for the way in, both recursive through `overflowing`.
///
/// The field names and their order are the declaration order of `Encoding` itself, deliberately:
/// `PyEncoding.__getstate__` pickles by serialising this, and a pickle written by an older build has
/// to keep loading. This is public API, not an internal format.
#[derive(Serialize)]
struct EncodingOut<'a> {
    ids: &'a [u32],
    type_ids: &'a [u32],
    tokens: &'a [String],
    words: &'a [Option<u32>],
    offsets: &'a [Offsets],
    special_tokens_mask: &'a [u32],
    attention_mask: &'a [u32],
    overflowing: Vec<EncodingOut<'a>>,
    sequence_ranges: &'a AHashMap<usize, Range<usize>>,
}

fn encoding_out(e: &Encoding) -> EncodingOut<'_> {
    EncodingOut {
        ids: e.get_ids(),
        type_ids: e.get_type_ids(),
        tokens: e.get_tokens(),
        words: e.get_word_ids(),
        offsets: e.get_offsets(),
        special_tokens_mask: e.get_special_tokens_mask(),
        attention_mask: e.get_attention_mask(),
        overflowing: e.get_overflowing().iter().map(encoding_out).collect(),
        sequence_ranges: e.get_sequence_ranges(),
    }
}

#[derive(Deserialize)]
struct EncodingIn {
    ids: Vec<u32>,
    type_ids: Vec<u32>,
    tokens: Vec<String>,
    words: Vec<Option<u32>>,
    offsets: Vec<Offsets>,
    special_tokens_mask: Vec<u32>,
    attention_mask: Vec<u32>,
    overflowing: Vec<EncodingIn>,
    sequence_ranges: AHashMap<usize, Range<usize>>,
}

impl EncodingIn {
    fn build(self) -> Encoding {
        Encoding::new(
            self.ids,
            self.type_ids,
            self.tokens,
            self.words,
            self.offsets,
            self.special_tokens_mask,
            self.attention_mask,
            self.overflowing.into_iter().map(Self::build).collect(),
            self.sequence_ranges,
        )
    }
}

pub mod encoding {
    use super::*;

    pub fn serialize<S: Serializer>(v: &Encoding, s: S) -> Result<S::Ok, S::Error> {
        encoding_out(v).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Encoding, D::Error> {
        Ok(EncodingIn::deserialize(d)?.build())
    }
}

/// A `Serialize`-able borrow of an [`Encoding`], for callers that need a value rather than a
/// `#[serde(with = ...)]` attribute — `PyEncoding.__getstate__` is the one that does.
pub struct EncodingRef<'a>(pub &'a Encoding);

impl Serialize for EncodingRef<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        encoding_out(self.0).serialize(serializer)
    }
}

/// An owning `Deserialize` newtype around [`Encoding`], the counterpart to [`EncodingRef`].
pub struct EncodingOwned(pub Encoding);

impl<'de> Deserialize<'de> for EncodingOwned {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self(EncodingIn::deserialize(deserializer)?.build()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Moved out of `tk-encode` with the serde it exercises: a `truncation` block written before
    /// `direction` existed still has to load, with `direction` defaulting to `Right`.
    #[test]
    fn test_deserialize_defaults() {
        let old_truncation_params = r#"{"max_length":256,"strategy":"LongestFirst","stride":0}"#;

        let params: TruncationParams =
            serde_json::from_str::<TruncationParamsMirror>(old_truncation_params)
                .unwrap()
                .into();

        assert_eq!(params.direction, TruncationDirection::Right);
    }

    /// `tk-encode` spells `SplitDelimiterBehavior`'s `Display` out by hand so the name survives a
    /// build with no serde in it. This is the test that stops the two from drifting; it lives here
    /// now because the serde half is here.
    #[test]
    fn display_matches_serde() {
        use tk_encode::tokenizer::normalizer::SplitDelimiterBehavior::*;

        #[derive(Serialize)]
        struct Wrap<'a>(#[serde(with = "SplitDelimiterBehaviorDef")] &'a SplitDelimiterBehavior);

        for behavior in [
            Removed,
            Isolated,
            MergedWithPrevious,
            MergedWithNext,
            Contiguous,
        ] {
            let via_serde = serde_json::to_string(&Wrap(&behavior)).unwrap();
            // `to_string` of a unit variant is a quoted string; `Display` is the bare name.
            assert_eq!(via_serde, format!("\"{behavior}\""));
        }
    }

    /// The pickle format is public API, so the mirror has to round-trip an `Encoding` unchanged --
    /// including the recursion through `overflowing` and the `sequence_ranges` map.
    #[test]
    fn encoding_round_trips() {
        let inner = Encoding::new(
            vec![7, 8],
            vec![0, 0],
            vec!["he".to_string(), "llo".to_string()],
            vec![Some(0), Some(0)],
            vec![(0, 2), (2, 5)],
            vec![0, 0],
            vec![1, 1],
            vec![],
            AHashMap::from_iter(vec![(0, 0..2)]),
        );
        let outer = Encoding::new(
            vec![1, 2, 3],
            vec![0, 0, 1],
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            vec![Some(0), Some(1), None],
            vec![(0, 1), (1, 2), (2, 3)],
            vec![0, 0, 1],
            vec![1, 1, 1],
            vec![inner],
            AHashMap::from_iter(vec![(0, 0..2), (1, 2..3)]),
        );

        let json = serde_json::to_string(&EncodingRef(&outer)).unwrap();
        let back: Encoding = serde_json::from_str::<EncodingOwned>(&json).unwrap().0;
        assert_eq!(outer, back);

        // The field names are the pickle format; assert them rather than trusting the derive.
        for field in [
            "ids",
            "type_ids",
            "tokens",
            "words",
            "offsets",
            "special_tokens_mask",
            "attention_mask",
            "overflowing",
            "sequence_ranges",
        ] {
            assert!(json.contains(&format!("\"{field}\"")), "missing {field}");
        }
    }
}
