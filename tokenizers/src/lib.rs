#![warn(clippy::all)]
#![doc(html_favicon_url = "https://huggingface.co/favicon.ico")]
#![doc(html_logo_url = "https://huggingface.co/landing/assets/huggingface_logo.svg")]

//! The core of `tokenizers`, written in Rust.
//! Provides an implementation of today's most used tokenizers, with a focus on performance and
//! versatility.
//!
//! # What is a Tokenizer
//!
//! A Tokenizer works as a pipeline, it processes some raw text as input and outputs an `Encoding`.
//! The various steps of the pipeline are:
//!
//! 1. The `Normalizer`: in charge of normalizing the text. Common examples of normalization are
//!    the [unicode normalization standards](https://unicode.org/reports/tr15/#Norm_Forms), such as `NFD` or `NFKC`.
//! 2. The `PreTokenizer`: in charge of creating initial words splits in the text. The most common way of
//!    splitting text is simply on whitespace.
//! 3. The `Model`: in charge of doing the actual tokenization. An example of a `Model` would be
//!    `BPE` or `WordPiece`.
//! 4. The `PostProcessor`: in charge of post-processing the `Encoding` to add anything relevant
//!    that, for example, a language model would need, such as special tokens.
//!
//! ## Deserialization and tokenization example
//!
//! ```no_run
//! use tokenizers::tokenizer::{Result, Tokenizer, EncodeInput};
//! use tokenizers::models::bpe::BPE;
//!
//! fn main() -> Result<()> {
//!     let bpe_builder = BPE::from_file("./path/to/vocab.json", "./path/to/merges.txt");
//!     let bpe = bpe_builder
//!         .dropout(0.1)
//!         .unk_token("[UNK]".into())
//!         .build()?;
//!
//!     let mut tokenizer = Tokenizer::new(bpe);
//!
//!     let encoding = tokenizer.encode("Hey there!", false)?;
//!     println!("{:?}", encoding.get_tokens());
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Training and serialization example
//!
//! ```no_run
//! use tokenizers::decoders::DecoderWrapper;
//! use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
//! use tokenizers::normalizers::{strip::Strip, unicode::NFC, utils::Sequence, NormalizerWrapper};
//! use tokenizers::pre_tokenizers::byte_level::ByteLevel;
//! use tokenizers::pre_tokenizers::PreTokenizerWrapper;
//! use tokenizers::processors::PostProcessorWrapper;
//! use tokenizers::{AddedToken, Model, Result, TokenizerBuilder};
//!
//! use std::path::Path;
//!
//! fn main() -> Result<()> {
//!     let vocab_size: usize = 100;
//!
//!     let trainer = BpeTrainerBuilder::new()
//!         .show_progress(true)
//!         .vocab_size(vocab_size)
//!         .min_frequency(0)
//!         .special_tokens(vec![
//!             AddedToken::from(String::from("<s>"), true),
//!             AddedToken::from(String::from("<pad>"), true),
//!             AddedToken::from(String::from("</s>"), true),
//!             AddedToken::from(String::from("<unk>"), true),
//!             AddedToken::from(String::from("<mask>"), true),
//!         ])
//!         .build();
//!
//!     let mut tokenizer = TokenizerBuilder::new()
//!         .with_model(BPE::default())
//!         .with_normalizer(Some(Sequence::new(vec![
//!             Strip::new(true, true).into(),
//!             NFC.into(),
//!         ])))
//!         .with_pre_tokenizer(Some(ByteLevel::default()))
//!         .with_post_processor(Some(ByteLevel::default()))
//!         .with_decoder(Some(ByteLevel::default()))
//!         .build()?;
//!
//!     let pretty = false;
//!     tokenizer
//!         .train(
//!             &trainer,
//!             vec!["path/to/vocab.txt".to_string()],
//!         )?
//!         .save("tokenizer.json", pretty)?;
//!
//!     Ok(())
//! }
//! ```
//!
//! # Additional information
//!
//! - tokenizers is designed to leverage CPU parallelism when possible. The level of parallelism is determined
//! by the total number of core/threads your CPU provides but this can be tuned by setting the `RAYON_RS_NUM_CPUS`
//! environment variable. As an example setting `RAYON_RS_NUM_CPUS=4` will allocate a maximum of 4 threads.
//! **_Please note this behavior may evolve in the future_**
//!
//! # Features
//! **progressbar**: The progress bar visualization is enabled by default. It might be disabled if
//!   compilation for certain targets is not supported by the [termios](https://crates.io/crates/termios)
//!   dependency of the [indicatif](https://crates.io/crates/indicatif) progress bar.

#[macro_use]
extern crate log;
#[macro_use]
extern crate lazy_static;

#[macro_use]
extern crate derive_builder;

#[macro_use]
pub mod utils;
pub mod decoders;
pub mod models;
pub mod normalizers;
pub mod pre_tokenizers;
pub mod processors;
pub mod tokenizer;

// Re-export from tokenizer
pub use tokenizer::*;

// Re-export also parallelism utils
pub use utils::parallelism;
