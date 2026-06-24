use tk_encode::models::bpe::BPE;
use tk_encode::models::unigram::Unigram;
use tk_encode::models::wordlevel::WordLevel;
use tk_encode::models::wordpiece::WordPiece;
use tk_encode::models::ModelWrapper;
use tk_encode::Model;

use crate::trainers::{
    BpeTrainer, TrainerWrapper, UnigramTrainer, WordLevelTrainer, WordPieceTrainer,
};
use crate::Trainer;

/// A `Model` that knows how to build a `Trainer` capable of training it.
///
/// In v1 this was the `type Trainer` / `get_trainer` part of the `Model` trait.
/// It now lives in `tk-train` so that `tk-encode` (inference) carries no
/// training-related coupling.
pub trait Trainable: Model {
    type Trainer: Trainer<Model = Self> + Sync;
    /// Get an instance of a Trainer capable of training this Model.
    fn get_trainer(&self) -> Self::Trainer;
}

impl Trainable for BPE {
    type Trainer = BpeTrainer;
    fn get_trainer(&self) -> BpeTrainer {
        BpeTrainer::default()
    }
}

impl Trainable for Unigram {
    type Trainer = UnigramTrainer;
    fn get_trainer(&self) -> UnigramTrainer {
        UnigramTrainer::default()
    }
}

impl Trainable for WordLevel {
    type Trainer = WordLevelTrainer;
    fn get_trainer(&self) -> WordLevelTrainer {
        WordLevelTrainer::default()
    }
}

impl Trainable for WordPiece {
    type Trainer = WordPieceTrainer;
    fn get_trainer(&self) -> WordPieceTrainer {
        WordPieceTrainer::builder().build()
    }
}

impl Trainable for ModelWrapper {
    type Trainer = TrainerWrapper;
    fn get_trainer(&self) -> TrainerWrapper {
        match self {
            Self::WordLevel(t) => t.get_trainer().into(),
            Self::WordPiece(t) => t.get_trainer().into(),
            Self::BPE(t) => t.get_trainer().into(),
            Self::Unigram(t) => t.get_trainer().into(),
        }
    }
}
