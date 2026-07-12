//! Popular tokenizer models.

pub mod bpe;
pub mod unigram;
pub mod wordlevel;
pub mod wordpiece;

use ahash::AHashMap;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error};
use serde_json::value::RawValue;

use crate::models::bpe::BPE;
use crate::models::unigram::Unigram;
use crate::models::wordlevel::WordLevel;
use crate::models::wordpiece::WordPiece;
use crate::{Model, Result, Token};

/// Wraps a vocab mapping (ID -> token) to a struct that will be serialized in order
/// of token ID, smallest to largest.
struct OrderedVocabIter<'a> {
    vocab_r: &'a AHashMap<u32, String>,
}

impl<'a> OrderedVocabIter<'a> {
    fn new(vocab_r: &'a AHashMap<u32, String>) -> Self {
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

#[derive(Serialize, Debug, PartialEq, Clone)]
#[serde(untagged)]
pub enum ModelWrapper {
    BPE(BPE),
    // WordPiece must stay before WordLevel here for deserialization (for retrocompatibility
    // with the versions not including the "type"), since WordLevel is a subset of WordPiece
    WordPiece(WordPiece),
    WordLevel(WordLevel),
    Unigram(Unigram),
}

impl<'de> Deserialize<'de> for ModelWrapper {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw: &'de RawValue = Deserialize::deserialize(deserializer)?;
        let model_json = raw.get();

        #[derive(Deserialize)]
        struct Tag {
            #[serde(rename = "type")]
            variant: Option<String>,
        }
        let tag: Tag = serde_json::from_str(model_json).map_err(D::Error::custom)?;

        match tag.variant.as_deref() {
            Some("BPE") => Ok(ModelWrapper::BPE(
                serde_json::from_str(model_json).map_err(D::Error::custom)?,
            )),
            Some("WordPiece") => Ok(ModelWrapper::WordPiece(
                serde_json::from_str(model_json).map_err(D::Error::custom)?,
            )),
            Some("WordLevel") => Ok(ModelWrapper::WordLevel(
                serde_json::from_str(model_json).map_err(D::Error::custom)?,
            )),
            Some("Unigram") => Ok(ModelWrapper::Unigram(
                serde_json::from_str(model_json).map_err(D::Error::custom)?,
            )),
            Some(other) => Err(D::Error::custom(format!("Unknown model type `{other}`"))),
            None => {
                if let Ok(m) = serde_json::from_str::<BPE>(model_json) {
                    Ok(ModelWrapper::BPE(m))
                } else if let Ok(m) = serde_json::from_str::<WordPiece>(model_json) {
                    Ok(ModelWrapper::WordPiece(m))
                } else if let Ok(m) = serde_json::from_str::<WordLevel>(model_json) {
                    Ok(ModelWrapper::WordLevel(m))
                } else if let Ok(m) = serde_json::from_str::<Unigram>(model_json) {
                    Ok(ModelWrapper::Unigram(m))
                } else {
                    Err(D::Error::custom("Model is not a known variant"))
                }
            }
        }
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

    fn save(&self, folder: &Path, name: Option<&str>) -> Result<Vec<PathBuf>> {
        match self {
            Self::WordLevel(t) => t.save(folder, name),
            Self::WordPiece(t) => t.save(folder, name),
            Self::BPE(t) => t.save(folder, name),
            Self::Unigram(t) => t.save(folder, name),
        }
    }
}

impl ModelWrapper {
    pub fn clear_cache(&mut self) {
        match self {
            Self::Unigram(model) => model.clear_cache(),
            Self::BPE(model) => model.clear_cache(),
            _ => (),
        }
    }
    pub fn resize_cache(&mut self, capacity: usize) {
        match self {
            Self::Unigram(model) => model.resize_cache(capacity),
            Self::BPE(model) => model.resize_cache(capacity),
            _ => (),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::bpe::{BpeBuilder, Vocab};

    #[test]
    fn incomplete_ordered_vocab() {
        let vocab_r: AHashMap<u32, String> =
            AHashMap::from([(0, "Hi".to_string()), (2, "There".to_string())]);

        let ordered = OrderedVocabIter::new(&vocab_r);

        let serialized = serde_json::to_string(&ordered).unwrap();
        assert_eq!(serialized, "{\"Hi\":0,\"There\":2}");
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
            Err(err) => assert!(
                err.to_string().starts_with("Merges text file invalid at line 1"),
                "unexpected error: {}",
                err
            ),
            _ => panic!("Expected an error here"),
        }
    }

    // The `type` tag must dispatch to the right variant for every model, and an
    // unknown tag must error (RawValue-based deserialization).
    #[test]
    fn model_wrapper_dispatches_on_type() {
        use crate::models::unigram::Unigram;
        use crate::models::wordlevel::WordLevel;
        use crate::models::wordpiece::WordPiece;

        // Round-trip each model's default through `ModelWrapper` and check the
        // resolved variant.
        let bpe = serde_json::to_string(&BPE::default()).unwrap();
        assert!(matches!(
            serde_json::from_str(&bpe).unwrap(),
            ModelWrapper::BPE(_)
        ));

        let wordpiece = serde_json::to_string(&WordPiece::default()).unwrap();
        assert!(matches!(
            serde_json::from_str(&wordpiece).unwrap(),
            ModelWrapper::WordPiece(_)
        ));

        let wordlevel = serde_json::to_string(&WordLevel::default()).unwrap();
        assert!(matches!(
            serde_json::from_str(&wordlevel).unwrap(),
            ModelWrapper::WordLevel(_)
        ));

        let unigram = serde_json::to_string(&Unigram::default()).unwrap();
        assert!(matches!(
            serde_json::from_str(&unigram).unwrap(),
            ModelWrapper::Unigram(_)
        ));

        let unknown = r#"{"type":"NotAModel","vocab":{"a":0}}"#;
        let err = serde_json::from_str::<ModelWrapper>(unknown).unwrap_err();
        assert!(err.to_string().starts_with("Unknown model type"));
    }
}
