//! Training half of the 🤗 Tokenizers library.
//!
//! This crate builds on top of [`tk_encode`] and provides everything related to
//! *training* a tokenizer: the [`Trainer`] trait, every concrete `*Trainer`, the
//! [`TrainerWrapper`] enum, the [`Trainable`] extension (the `get_trainer`
//! association that used to live on `tk_encode::Model`), and [`train_tokenizer`].
//!
//! [`train_tokenizer`] trains a fresh model through an existing pipeline's
//! normalization and pre-tokenization and returns a new pipeline retaining its
//! other stages. It is fresh training, not incremental training: the old model
//! vocabulary and old added tokens are not used to split the corpus or copied
//! into the result; only the trainer's special tokens are registered, so include
//! any specials retained settings need. Unigram byte fallback is rejected, and a
//! sequence longer than `u32::MAX` bytes after normalization is rejected before
//! pre-tokenization.
//!
//! To train a model directly without a pipeline, drive a trainer itself instead:
//! `feed`, then `train`.

#[cfg(not(any(
    feature = "bpe",
    feature = "unigram",
    feature = "wordpiece",
    feature = "wordlevel"
)))]
compile_error!("tk-train needs at least one model feature: bpe, unigram, wordpiece or wordlevel");

pub mod added_token_serde;
pub mod progress;
mod tokenizer;
mod trainable;
mod trainer;
pub mod trainers;

pub use progress::ProgressFormat;
pub use tokenizer::train_tokenizer;
pub use trainable::{ModelWrapper, Trainable};
pub use trainer::Trainer;
pub use trainers::*;
