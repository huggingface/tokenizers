//! Concrete model trainers and the [`TrainerWrapper`] enum that dispatches over them.

#[cfg(feature = "bpe")]
mod bpe;
#[cfg(feature = "unigram")]
mod unigram;
#[cfg(feature = "wordlevel")]
mod wordlevel;
#[cfg(feature = "wordpiece")]
mod wordpiece;

#[cfg(feature = "bpe")]
pub use bpe::*;
#[cfg(feature = "unigram")]
pub use unigram::*;
#[cfg(feature = "wordlevel")]
pub use wordlevel::*;
#[cfg(feature = "wordpiece")]
pub use wordpiece::*;

use serde::Deserialize;

use tk_encode::pipeline::PipelineModel;

use crate::error::Result;
use crate::trainer::{IntoPipelineModel, ModelTrainer, TrainingParams};

#[derive(Debug, Deserialize)]
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

impl ModelTrainer for TrainerWrapper {
    type Model = PipelineModel;

    fn train_model(&self, params: &TrainingParams) -> Result<PipelineModel> {
        match self {
            #[cfg(feature = "bpe")]
            Self::BpeTrainer(t) => t.train_model(params)?.into_pipeline_model(),
            #[cfg(feature = "wordpiece")]
            Self::WordPieceTrainer(t) => t.train_model(params)?.into_pipeline_model(),
            #[cfg(feature = "wordlevel")]
            Self::WordLevelTrainer(t) => t.train_model(params)?.into_pipeline_model(),
            #[cfg(feature = "unigram")]
            Self::UnigramTrainer(t) => t.train_model(params)?.into_pipeline_model(),
        }
    }

    fn check(&self, params: &TrainingParams) -> Result<()> {
        match self {
            #[cfg(feature = "bpe")]
            Self::BpeTrainer(t) => t.check(params),
            #[cfg(feature = "wordpiece")]
            Self::WordPieceTrainer(t) => t.check(params),
            #[cfg(feature = "wordlevel")]
            Self::WordLevelTrainer(t) => t.check(params),
            #[cfg(feature = "unigram")]
            Self::UnigramTrainer(t) => t.check(params),
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
            Self::BpeTrainer(t) => t.feed(iterator, process),
            #[cfg(feature = "wordpiece")]
            Self::WordPieceTrainer(t) => t.feed(iterator, process),
            #[cfg(feature = "wordlevel")]
            Self::WordLevelTrainer(t) => t.feed(iterator, process),
            #[cfg(feature = "unigram")]
            Self::UnigramTrainer(t) => t.feed(iterator, process),
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
