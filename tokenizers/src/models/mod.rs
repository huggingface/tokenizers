//! Popular tokenizer models.

pub mod bpe;
pub mod unigram;
pub mod wordlevel;
pub mod wordpiece;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::models::bpe::{BpeTrainer, BPE};
use crate::models::unigram::{Unigram, UnigramTrainer};
use crate::models::wordlevel::{WordLevel, WordLevelTrainer};
use crate::models::wordpiece::{WordPiece, WordPieceTrainer};
use crate::{AddedToken, Model, Result, Token, Trainer};

/// Wraps a vocab mapping (ID -> token) to a struct that will be serialized in order
/// of token ID, smallest to largest.
struct OrderedVocabIter<'a> {
    vocab_r: &'a HashMap<u32, String>,
}

impl<'a> OrderedVocabIter<'a> {
    fn new(vocab_r: &'a HashMap<u32, String>) -> Self {
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
        let result = if let Some(max) = self.vocab_r.iter().map(|(key, _)| key).max() {
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
            warn!("The OrderedVocab you are attempting to save contains holes for indices {:?}, your vocabulary could be corrupted !", holes);
            println!("The OrderedVocab you are attempting to save contains holes for indices {holes:?}, your vocabulary could be corrupted !");
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
        pub enum ModelUntagged {
            BPE(BPE),
            // WordPiece must stay before WordLevel here for deserialization (for retrocompatibility
            // with the versions not including the "type"), since WordLevel is a subset of WordPiece
            WordPiece(WordPiece),
            WordLevel(WordLevel),
            Unigram(Unigram),
        }

        let helper = ModelHelper::deserialize(deserializer)?;
        Ok(match helper {
            ModelHelper::Tagged(model) => match model.variant {
                EnumType::BPE => ModelWrapper::BPE(
                    serde_json::from_value(model.rest).map_err(serde::de::Error::custom)?,
                ),
                EnumType::WordPiece => ModelWrapper::WordPiece(
                    serde_json::from_value(model.rest).map_err(serde::de::Error::custom)?,
                ),
                EnumType::WordLevel => ModelWrapper::WordLevel(
                    serde_json::from_value(model.rest).map_err(serde::de::Error::custom)?,
                ),
                EnumType::Unigram => ModelWrapper::Unigram(
                    serde_json::from_value(model.rest).map_err(serde::de::Error::custom)?,
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
    type Trainer = TrainerWrapper;

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

    fn get_trainer(&self) -> Self::Trainer {
        match self {
            Self::WordLevel(t) => t.get_trainer().into(),
            Self::WordPiece(t) => t.get_trainer().into(),
            Self::BPE(t) => t.get_trainer().into(),
            Self::Unigram(t) => t.get_trainer().into(),
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

#[derive(Clone, Serialize, Deserialize)]
pub enum TrainerWrapper {
    BpeTrainer(BpeTrainer),
    WordPieceTrainer(WordPieceTrainer),
    WordLevelTrainer(WordLevelTrainer),
    UnigramTrainer(UnigramTrainer),
}

impl Trainer for TrainerWrapper {
    type Model = ModelWrapper;

    fn should_show_progress(&self) -> bool {
        match self {
            Self::BpeTrainer(bpe) => bpe.should_show_progress(),
            Self::WordPieceTrainer(wpt) => wpt.should_show_progress(),
            Self::WordLevelTrainer(wpt) => wpt.should_show_progress(),
            Self::UnigramTrainer(wpt) => wpt.should_show_progress(),
        }
    }

    fn train(&self, model: &mut ModelWrapper) -> Result<Vec<AddedToken>> {
        match self {
            Self::BpeTrainer(t) => match model {
                ModelWrapper::BPE(bpe) => t.train(bpe),
                _ => Err("BpeTrainer can only train a BPE".into()),
            },
            Self::WordPieceTrainer(t) => match model {
                ModelWrapper::WordPiece(wp) => t.train(wp),
                _ => Err("WordPieceTrainer can only train a WordPiece".into()),
            },
            Self::WordLevelTrainer(t) => match model {
                ModelWrapper::WordLevel(wl) => t.train(wl),
                _ => Err("WordLevelTrainer can only train a WordLevel".into()),
            },
            Self::UnigramTrainer(t) => match model {
                ModelWrapper::Unigram(u) => t.train(u),
                _ => Err("UnigramTrainer can only train a Unigram".into()),
            },
        }
    }

    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        match self {
            Self::BpeTrainer(bpe) => bpe.feed(iterator, process),
            Self::WordPieceTrainer(wpt) => wpt.feed(iterator, process),
            Self::WordLevelTrainer(wpt) => wpt.feed(iterator, process),
            Self::UnigramTrainer(wpt) => wpt.feed(iterator, process),
        }
    }
}

impl_enum_from!(BpeTrainer, TrainerWrapper, BpeTrainer);
impl_enum_from!(WordPieceTrainer, TrainerWrapper, WordPieceTrainer);
impl_enum_from!(UnigramTrainer, TrainerWrapper, UnigramTrainer);
impl_enum_from!(WordLevelTrainer, TrainerWrapper, WordLevelTrainer);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::bpe::{BpeBuilder, Vocab};

    #[test]
    fn trainer_wrapper_train_model_wrapper() {
        let trainer = TrainerWrapper::BpeTrainer(BpeTrainer::default());
        let mut model = ModelWrapper::Unigram(Unigram::default());

        let result = trainer.train(&mut model);
        assert!(result.is_err());
    }

    #[test]
    fn incomplete_ordered_vocab() {
        let vocab_r: HashMap<u32, String> =
            HashMap::from([(0, "Hi".to_string()), (2, "There".to_string())]);

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
            Err(err) => assert_eq!(err.to_string(), "Merges text file invalid at line 1"),
            _ => panic!("Expected an error here"),
        }
    }
}
