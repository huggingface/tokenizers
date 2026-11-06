//! Training half of the 🤗 Tokenizers library.
//!
//! This crate builds on top of [`tk_encode`] and provides everything related to
//! *training* a tokenizer: the [`Trainer`] trait, every concrete `*Trainer`, the
//! [`TrainerWrapper`] enum and the [`Trainable`] extension (the `get_trainer`
//! association that used to live on `tk_encode::Model`).
//!
//! There is no tokenizer-level `train` / `train_from_files` entry point: it was an
//! extension trait on `tk_convert`'s `TokenizerImpl`, and nothing in `tk-encode`
//! replaces that type -- the pipeline tokenizer hands out its model by shared
//! reference only. Drive a trainer directly instead: `feed`, then `train`.

#[cfg(not(any(
    feature = "bpe",
    feature = "unigram",
    feature = "wordpiece",
    feature = "wordlevel"
)))]
compile_error!("tk-train needs at least one model feature: bpe, unigram, wordpiece or wordlevel");

pub mod added_token_serde;
pub mod progress;
mod trainable;
mod trainer;
pub mod trainers;

pub use progress::ProgressFormat;
pub use trainable::{ModelWrapper, Trainable};
pub use trainer::Trainer;
pub use trainers::*;
