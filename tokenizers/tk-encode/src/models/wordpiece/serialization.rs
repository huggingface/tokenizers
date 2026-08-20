//! serde for the three models that stay in `tk-encode`: `WordPiece`, `WordLevel` and `Unigram`.
//!
//! Rust's orphan rule is the whole reason this file exists. Those three types are `tk-encode`'s and
//! `Serialize`/`Deserialize` are serde's, so this crate cannot implement one for the other. What it
//! *can* do is describe the JSON shape locally and convert — which is what
//! `#[serde(with = "mirror::…")]` on [`ModelWrapper`](super::ModelWrapper)'s variants asks for.
//!
//! `BPE` needs no mirror: it moved here wholesale, so it carries its serde directly in
//! [`bpe::serialization`](super::bpe::serialization).
//!
//! ## Why these are `mod`s and not `remote` derives
//!
//! All three were *hand-written* `Serialize`/`Deserialize` impls, not derives, and each hand-written
//! part is load-bearing:
//!
//! * the vocabulary goes out through [`OrderedVocabIter`], so `{token: id}` is written in id order
//!   rather than in hash order,
//! * `WordPiece` and `WordLevel` accumulate a *missing-fields* set and report the first one that is
//!   absent, which is what their `deserialization_should_fail` tests pin,
//! * every one of them builds the model through its builder or its `from` constructor, which is
//!   where the invariants (a reverse vocab that matches, a `min_score`, a trie) are established.
//!
//! So each is a `mod` with a `serialize`/`deserialize` pair holding the same visitor code as before.
//!
//! ## The `"type"` tag is optional for all three, and that is deliberate
//!
//! `ModelWrapper`'s legacy fallback is an *untagged* enum, so how lenient a variant is about the tag
//! decides which one claims a tag-less object. All three read `"type"` when it is present and
//! reject a wrong value, but none of them *require* it: `gpt2.json` and five other fixtures in
//! `data/` carry no tag at all. What separates them instead is their required fields, which is why
//! `WordPiece` has to stay ahead of `WordLevel` in every list — `WordLevel`'s fields are a subset of
//! `WordPiece`'s, so `WordLevel` would otherwise claim a tag-less `WordPiece`.

use ahash::{AHashMap, AHashSet};
use serde::de::{Error as _, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use tk_encode::models::unigram::Unigram;
use tk_encode::models::wordlevel::{WordLevel, WordLevelBuilder};
use tk_encode::models::wordpiece::{WordPiece, WordPieceBuilder};

use super::OrderedVocabIter;

// ---------------------------------------------------------------------------------------------
// WordPiece
// ---------------------------------------------------------------------------------------------

/// `#[serde(tag = ...)]` puts `"type"` first, which is where the hand-written impl's
/// `serialize_field("type", …)` put it. Small fields before the vocabulary, same as before.
#[derive(Serialize)]
#[serde(tag = "type", rename = "WordPiece")]
struct WordPieceOut<'a> {
    unk_token: &'a str,
    continuing_subword_prefix: &'a str,
    max_input_chars_per_word: usize,
    vocab: OrderedVocabIter<'a>,
}

pub mod wordpiece {
    use super::*;

    pub fn serialize<S: Serializer>(v: &WordPiece, s: S) -> Result<S::Ok, S::Error> {
        WordPieceOut {
            unk_token: &v.unk_token,
            continuing_subword_prefix: &v.continuing_subword_prefix,
            max_input_chars_per_word: v.max_input_chars_per_word,
            vocab: OrderedVocabIter::new(&v.vocab_r),
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<WordPiece, D::Error> {
        d.deserialize_struct(
            "WordPiece",
            &[
                "type",
                "unk_token",
                "continuing_subword_prefix",
                "max_input_chars_per_word",
                "vocab",
            ],
            WordPieceVisitor,
        )
    }
}

struct WordPieceVisitor;
impl<'de> Visitor<'de> for WordPieceVisitor {
    type Value = WordPiece;

    fn expecting(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "struct WordPiece")
    }

    fn visit_map<V>(self, mut map: V) -> std::result::Result<Self::Value, V::Error>
    where
        V: MapAccess<'de>,
    {
        let mut builder = WordPieceBuilder::new();
        let mut missing_fields = vec![
            // for retrocompatibility the "type" field is not mandatory
            "unk_token",
            "continuing_subword_prefix",
            "max_input_chars_per_word",
            "vocab",
        ]
        .into_iter()
        .collect::<AHashSet<_>>();

        while let Some(key) = map.next_key::<String>()? {
            match key.as_ref() {
                "unk_token" => builder = builder.unk_token(map.next_value()?),
                "continuing_subword_prefix" => {
                    builder = builder.continuing_subword_prefix(map.next_value()?)
                }
                "max_input_chars_per_word" => {
                    builder = builder.max_input_chars_per_word(map.next_value()?)
                }
                "vocab" => {
                    let vocab: AHashMap<String, u32> = map.next_value()?;
                    builder = builder.vocab(vocab)
                }
                "type" => match map.next_value()? {
                    "WordPiece" => {}
                    u => {
                        return Err(serde::de::Error::invalid_value(
                            serde::de::Unexpected::Str(u),
                            &"WordPiece",
                        ));
                    }
                },
                _ => {}
            }
            missing_fields.remove::<str>(&key);
        }

        if !missing_fields.is_empty() {
            Err(serde::de::Error::missing_field(
                missing_fields.iter().next().unwrap(),
            ))
        } else {
            Ok(builder.build().map_err(V::Error::custom)?)
        }
    }
}

// ---------------------------------------------------------------------------------------------
// WordLevel
// ---------------------------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(tag = "type", rename = "WordLevel")]
struct WordLevelOut<'a> {
    vocab: OrderedVocabIter<'a>,
    unk_token: &'a str,
}

pub mod wordlevel {
    use super::*;

    pub fn serialize<S: Serializer>(v: &WordLevel, s: S) -> Result<S::Ok, S::Error> {
        WordLevelOut {
            vocab: OrderedVocabIter::new(&v.vocab_r),
            unk_token: &v.unk_token,
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<WordLevel, D::Error> {
        d.deserialize_struct(
            "WordLevel",
            &["type", "vocab", "unk_token"],
            WordLevelVisitor,
        )
    }
}

struct WordLevelVisitor;
impl<'de> Visitor<'de> for WordLevelVisitor {
    type Value = WordLevel;

    fn expecting(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "struct WordLevel")
    }

    fn visit_map<V>(self, mut map: V) -> std::result::Result<Self::Value, V::Error>
    where
        V: MapAccess<'de>,
    {
        let mut builder = WordLevelBuilder::new();
        let mut missing_fields = vec![
            // for retrocompatibility the "type" field is not mandatory
            "unk_token",
            "vocab",
        ]
        .into_iter()
        .collect::<AHashSet<_>>();
        while let Some(key) = map.next_key::<String>()? {
            match key.as_ref() {
                "vocab" => builder = builder.vocab(map.next_value()?),
                "unk_token" => builder = builder.unk_token(map.next_value()?),
                "type" => match map.next_value()? {
                    "WordLevel" => {}
                    u => {
                        return Err(serde::de::Error::invalid_value(
                            serde::de::Unexpected::Str(u),
                            &"WordLevel",
                        ));
                    }
                },
                _ => {}
            }
            missing_fields.remove::<str>(&key);
        }

        if !missing_fields.is_empty() {
            Err(serde::de::Error::missing_field(
                missing_fields.iter().next().unwrap(),
            ))
        } else {
            Ok(builder.build().map_err(V::Error::custom)?)
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Unigram
// ---------------------------------------------------------------------------------------------

/// `unk_id` and the `(token, score)` table are read back through `Unigram`'s accessors: the fields
/// themselves are crate-private on the other side, and the score table is the one model vocabulary
/// that is a JSON *array* rather than an object, so it needs no ordering wrapper.
#[derive(Serialize)]
#[serde(tag = "type", rename = "Unigram")]
struct UnigramOut<'a> {
    unk_id: Option<usize>,
    vocab: &'a [(String, f64)],
    byte_fallback: bool,
}

pub mod unigram {
    use super::*;

    pub fn serialize<S: Serializer>(v: &Unigram, s: S) -> Result<S::Ok, S::Error> {
        UnigramOut {
            unk_id: v.unk_id(),
            vocab: v.vocab(),
            byte_fallback: v.byte_fallback(),
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Unigram, D::Error> {
        d.deserialize_struct(
            "Unigram",
            &["type", "vocab", "unk_id", "byte_fallback"],
            UnigramVisitor,
        )
    }
}

struct UnigramVisitor;
impl<'de> Visitor<'de> for UnigramVisitor {
    type Value = Unigram;

    fn expecting(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "struct Unigram")
    }

    fn visit_map<V>(self, mut map: V) -> std::result::Result<Self::Value, V::Error>
    where
        V: MapAccess<'de>,
    {
        let mut vocab: Option<Vec<(String, f64)>> = None;
        let mut unk_id: Option<usize> = None;
        let mut byte_fallback: bool = false;
        while let Some(key) = map.next_key::<String>()? {
            match key.as_ref() {
                "unk_id" => {
                    unk_id = map.next_value()?;
                }
                "byte_fallback" => byte_fallback = map.next_value()?,
                "vocab" => vocab = Some(map.next_value()?),
                "type" => match map.next_value()? {
                    "Unigram" => {}
                    u => {
                        return Err(serde::de::Error::invalid_value(
                            serde::de::Unexpected::Str(u),
                            &"Unigram",
                        ));
                    }
                },
                _ => (),
            }
        }
        match (vocab, unk_id, byte_fallback) {
            (Some(vocab), unk_id, byte_fallback) => Ok(Unigram::from(vocab, unk_id, byte_fallback)
                .map_err(|err| V::Error::custom(format!("Unable to load vocab {err:?}")))?),
            (None, _, _) => Err(V::Error::custom("Missing vocab")),
        }
    }
}

/// A whole `Unigram` as one serde item, for the two callers that need it outside a `ModelWrapper`:
/// [`super::unigram::load`] reads a bare `unigram.json`, and [`super::unigram::save`] writes one.
pub struct UnigramRef<'a>(pub &'a Unigram);

impl Serialize for UnigramRef<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        unigram::serialize(self.0, s)
    }
}

/// The `Deserialize` half of [`UnigramRef`]; a newtype because the model itself cannot carry the
/// impl. Unwrap with `.0`.
pub struct UnigramOwned(pub Unigram);

impl<'de> Deserialize<'de> for UnigramOwned {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Ok(Self(unigram::deserialize(d)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- WordPiece -------------------------------------------------------------------------
    //
    // The wrapper is what has the serde impls now, so a "just this model" round trip goes through
    // the mirror's own entry points rather than through `from_str::<WordPiece>`.

    fn wp_to_string(wp: &WordPiece) -> String {
        let mut out = Vec::new();
        wordpiece::serialize(wp, &mut serde_json::Serializer::new(&mut out)).unwrap();
        String::from_utf8(out).unwrap()
    }

    fn wp_from_str(s: &str) -> Result<WordPiece, serde_json::Error> {
        wordpiece::deserialize(&mut serde_json::Deserializer::from_str(s))
    }

    #[test]
    fn wordpiece_serde() {
        let wp = WordPiece::default();
        let wp_s = "{\
            \"type\":\"WordPiece\",\
            \"unk_token\":\"[UNK]\",\
            \"continuing_subword_prefix\":\"##\",\
            \"max_input_chars_per_word\":100,\
            \"vocab\":{}\
        }";

        assert_eq!(wp_to_string(&wp), wp_s);
        assert_eq!(wp_from_str(wp_s).unwrap(), wp);
    }

    #[test]
    fn wordpiece_deserialization_should_fail() {
        let missing_unk = "{\
            \"type\":\"WordPiece\",\
            \"continuing_subword_prefix\":\"##\",\
            \"max_input_chars_per_word\":100,\
            \"vocab\":{}\
        }";
        assert!(
            wp_from_str(missing_unk)
                .unwrap_err()
                .to_string()
                .starts_with("missing field `unk_token`")
        );

        let wrong_type = "{\
            \"type\":\"WordLevel\",\
            \"unk_token\":\"[UNK]\",\
            \"vocab\":{}\
        }";
        assert!(
            wp_from_str(wrong_type)
                .unwrap_err()
                .to_string()
                .starts_with("invalid value: string \"WordLevel\", expected WordPiece")
        );
    }

    // ---- WordLevel -------------------------------------------------------------------------

    fn wl_to_string(wl: &WordLevel) -> String {
        let mut out = Vec::new();
        wordlevel::serialize(wl, &mut serde_json::Serializer::new(&mut out)).unwrap();
        String::from_utf8(out).unwrap()
    }

    fn wl_from_str(s: &str) -> Result<WordLevel, serde_json::Error> {
        wordlevel::deserialize(&mut serde_json::Deserializer::from_str(s))
    }

    #[test]
    fn wordlevel_serde() {
        let wl = WordLevel::default();
        let wl_s = r#"{"type":"WordLevel","vocab":{},"unk_token":"<unk>"}"#;

        assert_eq!(wl_to_string(&wl), wl_s);
        assert_eq!(wl_from_str(wl_s).unwrap(), wl);
    }

    #[test]
    fn wordlevel_incomplete_vocab() {
        let vocab: AHashMap<String, u32> = [("<unk>".into(), 0), ("b".into(), 2)]
            .iter()
            .cloned()
            .collect();
        let wordlevel = WordLevelBuilder::default()
            .vocab(vocab)
            .unk_token("<unk>".to_string())
            .build()
            .unwrap();
        let wl_s = r#"{"type":"WordLevel","vocab":{"<unk>":0,"b":2},"unk_token":"<unk>"}"#;
        assert_eq!(wl_to_string(&wordlevel), wl_s);
        assert_eq!(wl_from_str(wl_s).unwrap(), wordlevel);
    }

    #[test]
    fn wordlevel_deserialization_should_fail() {
        let missing_unk = r#"{"type":"WordLevel","vocab":{}}"#;
        assert!(
            wl_from_str(missing_unk)
                .unwrap_err()
                .to_string()
                .starts_with("missing field `unk_token`")
        );

        let wrong_type = r#"{"type":"WordPiece","vocab":{}}"#;
        assert!(
            wl_from_str(wrong_type)
                .unwrap_err()
                .to_string()
                .starts_with("invalid value: string \"WordPiece\", expected WordLevel")
        );
    }

    // ---- Unigram ---------------------------------------------------------------------------

    fn unigram_round_trip(vocab: Vec<(String, f64)>, unk_id: Option<usize>) {
        let model = Unigram::from(vocab, unk_id, false).unwrap();
        let data = serde_json::to_string(&UnigramRef(&model)).unwrap();
        let reconstructed: UnigramOwned = serde_json::from_str(&data).unwrap();
        assert_eq!(model, reconstructed.0);
    }

    #[test]
    fn unigram_serialization() {
        unigram_round_trip(
            vec![("<unk>".to_string(), 0.0), ("a".to_string(), -0.5)],
            Some(0),
        );
    }

    #[test]
    fn unigram_serialization_unk_id_not_zero() {
        unigram_round_trip(
            vec![("a".to_string(), -0.5), ("<unk>".to_string(), 0.0)],
            Some(1),
        );
    }

    #[test]
    fn unigram_serialization_no_unk_id() {
        unigram_round_trip(vec![("a".to_string(), -0.5)], None);
    }
}
