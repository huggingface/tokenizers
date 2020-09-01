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
use crate::models::wordlevel::WordLevel;
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
        let iter = (0u32..(self.vocab_r.len() as u32)).map(|i| (&self.vocab_r[&i], i));
        serializer.collect_map(iter)
    }
}

#[derive(Deserialize, Serialize, Debug, PartialEq, Clone)]
#[serde(untagged)]
pub enum ModelWrapper {
    WordPiece(WordPiece),
    BPE(BPE),
    WordLevel(WordLevel),
    Unigram(Unigram),
}

impl_enum_from!(WordLevel, ModelWrapper, WordLevel);
impl_enum_from!(WordPiece, ModelWrapper, WordPiece);
impl_enum_from!(BPE, ModelWrapper, BPE);
impl_enum_from!(Unigram, ModelWrapper, Unigram);

impl Model for ModelWrapper {
    fn tokenize(&self, tokens: &str) -> Result<Vec<Token>> {
        use ModelWrapper::*;
        match self {
            WordLevel(t) => t.tokenize(tokens),
            WordPiece(t) => t.tokenize(tokens),
            BPE(t) => t.tokenize(tokens),
            Unigram(t) => t.tokenize(tokens),
        }
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        use ModelWrapper::*;
        match self {
            WordLevel(t) => t.token_to_id(token),
            WordPiece(t) => t.token_to_id(token),
            BPE(t) => t.token_to_id(token),
            Unigram(t) => t.token_to_id(token),
        }
    }

    fn id_to_token(&self, id: u32) -> Option<&str> {
        use ModelWrapper::*;
        match self {
            WordLevel(t) => t.id_to_token(id),
            WordPiece(t) => t.id_to_token(id),
            BPE(t) => t.id_to_token(id),
            Unigram(t) => t.id_to_token(id),
        }
    }

    fn get_vocab(&self) -> &HashMap<String, u32> {
        use ModelWrapper::*;
        match self {
            WordLevel(t) => t.get_vocab(),
            WordPiece(t) => t.get_vocab(),
            BPE(t) => t.get_vocab(),
            Unigram(t) => t.get_vocab(),
        }
    }

    fn get_vocab_size(&self) -> usize {
        use ModelWrapper::*;
        match self {
            WordLevel(t) => t.get_vocab_size(),
            WordPiece(t) => t.get_vocab_size(),
            BPE(t) => t.get_vocab_size(),
            Unigram(t) => t.get_vocab_size(),
        }
    }

    fn save(&self, folder: &Path, name: Option<&str>) -> Result<Vec<PathBuf>> {
        use ModelWrapper::*;
        match self {
            WordLevel(t) => t.save(folder, name),
            WordPiece(t) => t.save(folder, name),
            BPE(t) => t.save(folder, name),
            Unigram(t) => t.save(folder, name),
        }
    }
}

pub enum TrainerWrapper {
    BpeTrainer(BpeTrainer),
    WordPieceTrainer(WordPieceTrainer),
    UnigramTrainer(UnigramTrainer),
}

impl Trainer for TrainerWrapper {
    type Model = ModelWrapper;

    fn should_show_progress(&self) -> bool {
        match self {
            TrainerWrapper::BpeTrainer(bpe) => bpe.should_show_progress(),
            TrainerWrapper::WordPieceTrainer(wpt) => wpt.should_show_progress(),
            TrainerWrapper::UnigramTrainer(wpt) => wpt.should_show_progress(),
        }
    }

    fn train(&self, words: HashMap<String, u32>) -> Result<(Self::Model, Vec<AddedToken>)> {
        match self {
            TrainerWrapper::BpeTrainer(bpe) => bpe.train(words).map(|(m, t)| (m.into(), t)),
            TrainerWrapper::WordPieceTrainer(wpt) => wpt.train(words).map(|(m, t)| (m.into(), t)),
            TrainerWrapper::UnigramTrainer(wpt) => wpt.train(words).map(|(m, t)| (m.into(), t)),
        }
    }

    fn process_tokens(&self, words: &mut HashMap<String, u32>, tokens: Vec<String>) {
        match self {
            TrainerWrapper::BpeTrainer(bpe) => bpe.process_tokens(words, tokens),
            TrainerWrapper::WordPieceTrainer(wpt) => wpt.process_tokens(words, tokens),
            TrainerWrapper::UnigramTrainer(wpt) => wpt.process_tokens(words, tokens),
        }
    }
}

impl_enum_from!(BpeTrainer, TrainerWrapper, BpeTrainer);
impl_enum_from!(WordPieceTrainer, TrainerWrapper, WordPieceTrainer);
impl_enum_from!(UnigramTrainer, TrainerWrapper, UnigramTrainer);
