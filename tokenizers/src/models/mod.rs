//! Popular tokenizer models.

pub mod bpe;
pub mod unigram;
pub mod wordlevel;
pub mod wordpiece;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize, Serializer};

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

impl<'a> Serialize for OrderedVocabIter<'a> {
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
            println!("The OrderedVocab you are attempting to save contains holes for indices {:?}, your vocabulary could be corrupted !", holes);
        }
        result
    }
}

#[derive(Deserialize, Serialize, Debug, PartialEq, Clone)]
#[serde(untagged)]
pub enum ModelWrapper {
    BPE(BPE),
    // WordPiece must stay before WordLevel here for deserialization (for retrocompatibility
    // with the versions not including the "type"), since WordLevel is a subset of WordPiece
    WordPiece(WordPiece),
    WordLevel(WordLevel),
    Unigram(Unigram),
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
}
