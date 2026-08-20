//! `ModelWrapper`: the `model` field of a `tokenizer.json`, as an enum over every model.
//!
//! The `Deserialize` impl is hand-written for two reasons that are both backwards compatibility:
//! configs written before the `"type"` tag existed have to keep loading (the `Legacy` arm, an
//! untagged fallback whose *variant order* decides ties — see the comment on `ModelUntagged`), and
//! a tagged config must be routed by its tag rather than by shape.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ahash::AHashMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use tk_encode::models::unigram::Unigram;
use tk_encode::models::wordlevel::WordLevel;
use tk_encode::models::wordpiece::WordPiece;
use tk_encode::{Model, Result, Token};

use crate::macros::impl_enum_from;

pub mod bpe;
pub mod mirror;
pub mod unigram;
pub mod wordlevel;
pub mod wordpiece;

pub use bpe::BPE;

/// `BPE` is this crate's own type, so it carries its serde directly (see [`bpe::serialization`]).
/// The other three are `tk-encode`'s, and a foreign crate cannot implement `Serialize` for a foreign
/// type — so each names a mirror of its on-disk shape in [`mirror`].
#[derive(Debug, PartialEq, Clone, Serialize)]
#[serde(untagged)]
// BPE is the big variant (~472 B of merge/fold state) and the only one that is always
// compiled, so any two-model combination trips the size-difference lint. Boxing it would
// put an indirection on the hot model, which is the opposite of what we want.
#[allow(clippy::large_enum_variant)]
pub enum ModelWrapper {
    BPE(BPE),
    // WordPiece must stay before WordLevel here for deserialization (for retrocompatibility
    // with the versions not including the "type"), since WordLevel is a subset of WordPiece
    #[serde(with = "mirror::wordpiece")]
    WordPiece(WordPiece),
    #[serde(with = "mirror::wordlevel")]
    WordLevel(WordLevel),
    #[serde(with = "mirror::unigram")]
    Unigram(Unigram),
}

/// Wraps a vocab mapping (ID -> token) to a struct that will be serialized in order
/// of token ID, smallest to largest.
///
/// Moved here from `tk-encode` with the `Serialize` impls that are its only callers: it exists to
/// give a `{token: id}` object a *deterministic* order, which is a property of the file being
/// written, not of the model doing the encoding.
pub(crate) struct OrderedVocabIter<'a> {
    vocab_r: &'a AHashMap<u32, String>,
}

impl<'a> OrderedVocabIter<'a> {
    pub(crate) fn new(vocab_r: &'a AHashMap<u32, String>) -> Self {
        Self { vocab_r }
    }
}

impl Serialize for OrderedVocabIter<'_> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // There could be holes so max + 1 is more correct than vocab_r.len()
        let mut holes = vec![];
        let result = if let Some(max) = self.vocab_r.keys().max() {
            let iter = (0..*max + 1).filter_map(|i| {
                if let Some(token) = self.vocab_r.get(&i) {
                    Some((token, i))
                } else {
                    holes.push(i);
                    None
                }
            });
            serializer.collect_map(iter)
        } else {
            serializer.collect_map(std::iter::empty::<(&str, u32)>())
        };

        if !holes.is_empty() {
            warn!(
                "The OrderedVocab you are attempting to serialize contains holes for indices {holes:?}, your vocabulary could be corrupted!"
            );
        }
        result
    }
}

impl<'de> Deserialize<'de> for ModelWrapper {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        pub struct Tagged {
            #[serde(rename = "type")]
            variant: EnumType,
            #[serde(flatten)]
            rest: serde_json::Value,
        }
        #[derive(Deserialize)]
        pub enum EnumType {
            BPE,
            WordPiece,
            WordLevel,
            Unigram,
        }

        #[derive(Deserialize)]
        #[serde(untagged)]
        pub enum ModelHelper {
            Tagged(Tagged),
            Legacy(serde_json::Value),
        }

        #[derive(Deserialize)]
        #[serde(untagged)]
        #[allow(clippy::large_enum_variant)]
        pub enum ModelUntagged {
            BPE(BPE),
            // WordPiece must stay before WordLevel here for deserialization (for retrocompatibility
            // with the versions not including the "type"), since WordLevel is a subset of WordPiece
            #[serde(with = "mirror::wordpiece")]
            WordPiece(WordPiece),
            #[serde(with = "mirror::wordlevel")]
            WordLevel(WordLevel),
            #[serde(with = "mirror::unigram")]
            Unigram(Unigram),
        }

        let helper = ModelHelper::deserialize(deserializer)?;
        Ok(match helper {
            ModelHelper::Tagged(model) => match model.variant {
                EnumType::BPE => ModelWrapper::BPE(
                    serde_json::from_value(model.rest).map_err(serde::de::Error::custom)?,
                ),
                // `model.rest` deliberately does *not* get its `"type"` key put back: the three
                // model mirrors treat the tag as optional, exactly as the impls they replaced did,
                // and re-inserting it would only make the tagged path stricter than the untagged
                // one it shares its visitors with.
                EnumType::WordPiece => ModelWrapper::WordPiece(
                    mirror::wordpiece::deserialize(model.rest).map_err(serde::de::Error::custom)?,
                ),
                EnumType::WordLevel => ModelWrapper::WordLevel(
                    mirror::wordlevel::deserialize(model.rest).map_err(serde::de::Error::custom)?,
                ),
                EnumType::Unigram => ModelWrapper::Unigram(
                    mirror::unigram::deserialize(model.rest).map_err(serde::de::Error::custom)?,
                ),
            },
            ModelHelper::Legacy(value) => {
                let untagged = serde_json::from_value(value).map_err(serde::de::Error::custom)?;
                match untagged {
                    ModelUntagged::BPE(bpe) => ModelWrapper::BPE(bpe),
                    ModelUntagged::WordPiece(bpe) => ModelWrapper::WordPiece(bpe),
                    ModelUntagged::WordLevel(bpe) => ModelWrapper::WordLevel(bpe),
                    ModelUntagged::Unigram(bpe) => ModelWrapper::Unigram(bpe),
                }
            }
        })
    }
}

impl_enum_from!(WordLevel, ModelWrapper, WordLevel);
impl_enum_from!(WordPiece, ModelWrapper, WordPiece);
impl_enum_from!(BPE, ModelWrapper, BPE);
impl_enum_from!(Unigram, ModelWrapper, Unigram);

impl Model for ModelWrapper {
    fn tokenize(&self, tokens: &str) -> Result<Vec<Token>> {
        match self {
            Self::WordLevel(t) => t.tokenize(tokens),
            Self::WordPiece(t) => t.tokenize(tokens),
            Self::BPE(t) => t.tokenize(tokens),
            Self::Unigram(t) => t.tokenize(tokens),
        }
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        match self {
            Self::WordLevel(t) => t.token_to_id(token),
            Self::WordPiece(t) => t.token_to_id(token),
            Self::BPE(t) => t.token_to_id(token),
            Self::Unigram(t) => t.token_to_id(token),
        }
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        match self {
            Self::WordLevel(t) => t.id_to_token(id),
            Self::WordPiece(t) => t.id_to_token(id),
            Self::BPE(t) => t.id_to_token(id),
            Self::Unigram(t) => t.id_to_token(id),
        }
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        match self {
            Self::WordLevel(t) => t.get_vocab(),
            Self::WordPiece(t) => t.get_vocab(),
            Self::BPE(t) => t.get_vocab(),
            Self::Unigram(t) => t.get_vocab(),
        }
    }

    fn get_vocab_size(&self) -> usize {
        match self {
            Self::WordLevel(t) => t.get_vocab_size(),
            Self::WordPiece(t) => t.get_vocab_size(),
            Self::BPE(t) => t.get_vocab_size(),
            Self::Unigram(t) => t.get_vocab_size(),
        }
    }

    /// `WordLevel` and `Unigram` are written here rather than delegated.
    ///
    /// Both of their files are *serde* output — a `vocab.json` through [`OrderedVocabIter`] and a
    /// pretty-printed whole-model `unigram.json` — and both types live in `tk-encode`, which links
    /// no serde. `Model::save` is a trait method, so their impls cannot move here with the writing
    /// (the orphan rule: foreign trait, foreign type), and hand-rolling a byte-identical
    /// `serde_json` writer over `f64` scores is precisely the kind of thing that drifts silently.
    /// So the write happens on this side, at the one place every real caller passes through:
    /// `Tokenizer::save_model`, python's `Model.save` and node's all hold a `ModelWrapper`.
    /// `tk-encode`'s own impls report that this is the config layer's job.
    fn save(&self, folder: &Path, name: Option<&str>) -> Result<Vec<PathBuf>> {
        match self {
            Self::WordLevel(t) => wordlevel::save(t, folder, name),
            Self::WordPiece(t) => t.save(folder, name),
            Self::BPE(t) => t.save(folder, name),
            Self::Unigram(t) => unigram::save(t, folder, name),
        }
    }
}

impl ModelWrapper {
    pub fn clear_cache(&mut self) {
        match self {
            Self::Unigram(model) => model.clear_cache(),
            Self::BPE(model) => model.clear_cache(),
            // BPE and Unigram both have explicit arms, so only these two can reach the catch-all.
            _ => (),
        }
    }
    pub fn resize_cache(&mut self, capacity: usize) {
        match self {
            Self::Unigram(model) => model.resize_cache(capacity),
            Self::BPE(model) => model.resize_cache(capacity),
            // BPE and Unigram both have explicit arms, so only these two can reach the catch-all.
            _ => (),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::bpe::BpeBuilder;
    use tk_encode::models::bpe::Vocab;

    #[test]
    fn incomplete_ordered_vocab() {
        let vocab_r: AHashMap<u32, String> =
            AHashMap::from([(0, "Hi".to_string()), (2, "There".to_string())]);

        let ordered = OrderedVocabIter::new(&vocab_r);

        let serialized = serde_json::to_string(&ordered).unwrap();
        assert_eq!(serialized, "{\"Hi\":0,\"There\":2}");
    }

    #[test]
    fn ordered_vocab_iter() {
        let vocab_r: AHashMap<u32, String> = [
            (0, "a".into()),
            (1, "b".into()),
            (2, "c".into()),
            (3, "ab".into()),
        ]
        .iter()
        .cloned()
        .collect();
        let order_vocab_iter = OrderedVocabIter::new(&vocab_r);
        let serialized = serde_json::to_string(&order_vocab_iter).unwrap();
        assert_eq!(serialized, "{\"a\":0,\"b\":1,\"c\":2,\"ab\":3}");
    }

    #[test]
    fn serialization() {
        let vocab: Vocab = [
            ("<unk>".into(), 0),
            ("a".into(), 1),
            ("b".into(), 2),
            ("ab".into(), 3),
        ]
        .iter()
        .cloned()
        .collect();
        let bpe = BpeBuilder::default()
            .vocab_and_merges(vocab, vec![("a".to_string(), "b".to_string())])
            .unk_token("<unk>".to_string())
            .ignore_merges(true)
            .build()
            .unwrap();

        let model = ModelWrapper::BPE(bpe);
        let legacy = r#"{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b":2,"ab":3},"merges":["a b"]}"#;
        let legacy = serde_json::from_str(legacy).unwrap();
        assert_eq!(model, legacy);

        let data = serde_json::to_string(&model).unwrap();
        assert_eq!(
            data,
            r#"{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b":2,"ab":3},"merges":[["a","b"]]}"#
        );
        let reconstructed = serde_json::from_str(&data).unwrap();
        assert_eq!(model, reconstructed);

        // Legacy check, type is not necessary.
        let legacy = r#"{"dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b":2,"ab":3},"merges":["a b"]}"#;
        let reconstructed = serde_json::from_str(legacy).unwrap();
        assert_eq!(model, reconstructed);

        let invalid = r#"{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b":2,"ab":3},"merges":["a b c"]}"#;
        let reconstructed: std::result::Result<ModelWrapper, serde_json::Error> =
            serde_json::from_str(invalid);
        match reconstructed {
            Err(err) => assert_eq!(err.to_string(), "Merges text file invalid at line 1"),
            _ => panic!("Expected an error here"),
        }
    }
}
