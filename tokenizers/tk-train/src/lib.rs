//! Training half of the 🤗 Tokenizers library.
//!
//! This crate builds on top of [`tk_encode`] and provides everything related to
//! *training* a tokenizer: `TokenizerTrainer`, the entry point, the [`ModelTrainer`]
//! trait and every concrete `*Trainer`, and the [`TrainerWrapper`] enum that dispatches
//! over them.
//!
//! A trainer learns a model from the words the rest of the pipeline produces.
//! [`TokenizerTrainer`] holds both, checks that they agree, runs the text through the
//! pipeline, trains, and returns the finished tokenizer:
//!
//! ```no_run
//! use tk_encode::pipeline::PipelineTokenizer;
//! use tk_train::{TokenizerTrainerBuilder, TrainingError};
//!
//! fn retrain(tokenizer: &PipelineTokenizer, lines: &[&str]) -> Result<PipelineTokenizer, TrainingError> {
//!     TokenizerTrainerBuilder::from_tokenizer(tokenizer)?
//!         .vocab_size(32_000)
//!         .build()?
//!         .train(lines.iter())
//! }
//! ```

#[cfg(not(any(
    feature = "bpe",
    feature = "unigram",
    feature = "wordpiece",
    feature = "wordlevel"
)))]
compile_error!("tk-train needs at least one model feature: bpe, unigram, wordpiece or wordlevel");

#[cfg(feature = "parity-aware-bpe")]
mod added_token_serde;
mod error;
mod progress;
mod trainer;
mod trainer_builder;
mod trainers;

pub use error::TrainingError;
pub use progress::ProgressFormat;
pub use trainer::{
    IntoPipelineModel, ModelTrainer, PaddingSpec, PostProcessorSpec, TemplateSpec,
    TokenizerTrainer, TrainingParams,
};
pub use trainer_builder::TokenizerTrainerBuilder;
pub use trainers::TrainerWrapper;
#[cfg(feature = "bpe")]
pub use trainers::{BpeTrainer, BpeTrainerBuilder};
#[cfg(feature = "parity-aware-bpe")]
pub use trainers::{ParityBpeTrainer, ParityBpeTrainerBuilder, ParityVariant};
#[cfg(feature = "unigram")]
pub use trainers::{UnigramTrainer, UnigramTrainerBuilder};
#[cfg(feature = "wordlevel")]
pub use trainers::{WordLevelTrainer, WordLevelTrainerBuilder};
#[cfg(feature = "wordpiece")]
pub use trainers::{WordPieceTrainer, WordPieceTrainerBuilder};
