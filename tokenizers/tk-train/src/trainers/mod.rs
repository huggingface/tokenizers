//! Concrete model trainers and the [`TrainerWrapper`] enum that dispatches over
//! them, paired with [`crate::ModelWrapper`] on the model side.

#[cfg(feature = "bpe")]
pub mod bpe;
#[cfg(feature = "unigram")]
pub mod unigram;
#[cfg(feature = "wordlevel")]
pub mod wordlevel;
#[cfg(feature = "wordpiece")]
pub mod wordpiece;

#[cfg(feature = "bpe")]
pub use bpe::*;
#[cfg(feature = "unigram")]
pub use unigram::*;
#[cfg(feature = "wordlevel")]
pub use wordlevel::*;
#[cfg(feature = "wordpiece")]
pub use wordpiece::*;

use serde::{Deserialize, Serialize};

use crate::ModelWrapper;
use tk_encode::Result;
use tk_encode::vocab::bucket_added_vocabulary::AddedToken;

use crate::Trainer;

#[derive(Clone, Serialize, Deserialize)]
pub enum TrainerWrapper {
    #[cfg(feature = "bpe")]
    BpeTrainer(BpeTrainer),
    #[cfg(feature = "wordpiece")]
    WordPieceTrainer(WordPieceTrainer),
    #[cfg(feature = "wordlevel")]
    WordLevelTrainer(WordLevelTrainer),
    #[cfg(feature = "unigram")]
    UnigramTrainer(UnigramTrainer),
}

impl Trainer for TrainerWrapper {
    type Model = ModelWrapper;

    fn should_show_progress(&self) -> bool {
        match self {
            #[cfg(feature = "bpe")]
            Self::BpeTrainer(bpe) => bpe.should_show_progress(),
            #[cfg(feature = "wordpiece")]
            Self::WordPieceTrainer(wpt) => wpt.should_show_progress(),
            #[cfg(feature = "wordlevel")]
            Self::WordLevelTrainer(wpt) => wpt.should_show_progress(),
            #[cfg(feature = "unigram")]
            Self::UnigramTrainer(wpt) => wpt.should_show_progress(),
        }
    }

    // With a single model compiled in, the `_` arms can never match.
    #[allow(unreachable_patterns)]
    fn train(&self, model: &mut ModelWrapper) -> Result<Vec<AddedToken>> {
        match self {
            #[cfg(feature = "bpe")]
            Self::BpeTrainer(t) => match model {
                ModelWrapper::BPE(bpe) => t.train(bpe),
                _ => Err("BpeTrainer can only train a BPE".into()),
            },
            #[cfg(feature = "wordpiece")]
            Self::WordPieceTrainer(t) => match model {
                ModelWrapper::WordPiece(wp) => t.train(wp),
                _ => Err("WordPieceTrainer can only train a WordPiece".into()),
            },
            #[cfg(feature = "wordlevel")]
            Self::WordLevelTrainer(t) => match model {
                ModelWrapper::WordLevel(wl) => t.train(wl),
                _ => Err("WordLevelTrainer can only train a WordLevel".into()),
            },
            #[cfg(feature = "unigram")]
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
            #[cfg(feature = "bpe")]
            Self::BpeTrainer(bpe) => bpe.feed(iterator, process),
            #[cfg(feature = "wordpiece")]
            Self::WordPieceTrainer(wpt) => wpt.feed(iterator, process),
            #[cfg(feature = "wordlevel")]
            Self::WordLevelTrainer(wpt) => wpt.feed(iterator, process),
            #[cfg(feature = "unigram")]
            Self::UnigramTrainer(wpt) => wpt.feed(iterator, process),
        }
    }
}

#[cfg(feature = "bpe")]
impl From<BpeTrainer> for TrainerWrapper {
    fn from(t: BpeTrainer) -> Self {
        Self::BpeTrainer(t)
    }
}
#[cfg(feature = "wordpiece")]
impl From<WordPieceTrainer> for TrainerWrapper {
    fn from(t: WordPieceTrainer) -> Self {
        Self::WordPieceTrainer(t)
    }
}
#[cfg(feature = "unigram")]
impl From<UnigramTrainer> for TrainerWrapper {
    fn from(t: UnigramTrainer) -> Self {
        Self::UnigramTrainer(t)
    }
}
#[cfg(feature = "wordlevel")]
impl From<WordLevelTrainer> for TrainerWrapper {
    fn from(t: WordLevelTrainer) -> Self {
        Self::WordLevelTrainer(t)
    }
}

#[cfg(test)]
#[cfg(all(feature = "bpe", feature = "unigram"))]
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
