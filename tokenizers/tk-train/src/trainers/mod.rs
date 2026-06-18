//! Concrete model trainers and the [`TrainerWrapper`] enum that dispatches over
//! them (mirroring `tk_encode::models::ModelWrapper`).

pub mod bpe;
pub mod unigram;
pub mod wordlevel;
pub mod wordpiece;

pub use bpe::*;
pub use unigram::*;
pub use wordlevel::*;
pub use wordpiece::*;

use serde::{Deserialize, Serialize};

use tk_encode::models::ModelWrapper;
use tk_encode::{AddedToken, Result};

use crate::Trainer;

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

impl From<BpeTrainer> for TrainerWrapper {
    fn from(t: BpeTrainer) -> Self {
        Self::BpeTrainer(t)
    }
}
impl From<WordPieceTrainer> for TrainerWrapper {
    fn from(t: WordPieceTrainer) -> Self {
        Self::WordPieceTrainer(t)
    }
}
impl From<UnigramTrainer> for TrainerWrapper {
    fn from(t: UnigramTrainer) -> Self {
        Self::UnigramTrainer(t)
    }
}
impl From<WordLevelTrainer> for TrainerWrapper {
    fn from(t: WordLevelTrainer) -> Self {
        Self::WordLevelTrainer(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tk_encode::models::unigram::Unigram;

    #[test]
    fn trainer_wrapper_train_model_wrapper() {
        let trainer = TrainerWrapper::BpeTrainer(BpeTrainer::default());
        let mut model = ModelWrapper::Unigram(Unigram::default());

        let result = trainer.train(&mut model);
        assert!(result.is_err());
    }
}
