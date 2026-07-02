#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(clippy::all)]
#![allow(clippy::upper_case_acronyms)]
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
//!    More details about how to use the `Normalizers` are available on the
//!    [Hugging Face blog](https://huggingface.co/docs/tokenizers/components#normalizers)
//! 2. The `PreTokenizer`: in charge of creating initial words splits in the text. The most common way of
//!    splitting text is simply on whitespace.
//! 3. The `Model`: in charge of doing the actual tokenization. An example of a `Model` would be
//!    `BPE` or `WordPiece`.
//! 4. The `PostProcessor`: in charge of post-processing the `Encoding` to add anything relevant
//!    that, for example, a language model would need, such as special tokens.
//!
//! ## Loading a pretrained tokenizer from the Hub
//! ```
//! use tk_encode::tokenizer::{Result, Tokenizer};
//!
//! fn main() -> Result<()> {
//!     # #[cfg(feature = "http")]
//!     # {
//!     // needs http feature enabled
//!     let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None)?;
//!
//!     let encoding = tokenizer.encode("Hey there!", false)?;
//!     println!("{:?}", encoding.get_tokens());
//!     # }
//!     Ok(())
//! }
//! ```
//!
//! ## Deserialization and tokenization example
//!
//! ```no_run
//! use tk_encode::tokenizer::{Result, Tokenizer, EncodeInput};
//! use tk_encode::models::bpe::BPE;
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
//! Training lives in the companion `tk-train` crate (re-exported by the
//! `tokenizers` umbrella crate behind the `train` feature).
//!
//! # Additional information
//!
//! - tokenizers is designed to leverage CPU parallelism when possible. The level of parallelism is determined
//!   by the total number of core/threads your CPU provides but this can be tuned by setting the `RAYON_RS_NUM_THREADS`
//!   environment variable. As an example setting `RAYON_RS_NUM_THREADS=4` will allocate a maximum of 4 threads.
//!   **_Please note this behavior may evolve in the future_**
//!
//! # Features
//!
//! - **progressbar**: The progress bar visualization is enabled by default. It might be disabled if
//!   compilation for certain targets is not supported by the [termios](https://crates.io/crates/termios)
//!   dependency of the [indicatif](https://crates.io/crates/indicatif) progress bar.
//!
//! - **http**: This feature enables downloading the tokenizer via HTTP. It is disabled by default.
//!   With this feature enabled, `Tokenizer::from_pretrained` becomes accessible.

#[macro_use]
extern crate log;

#[macro_use]
extern crate derive_builder;

#[macro_use]
pub mod utils;
pub mod added_vocabulary;
pub mod decoders;
pub mod models;
pub mod normalizers;
pub mod pre_tokenizers;
pub mod processors;
pub mod tokenizer;

// Re-export from tokenizer
pub use tokenizer::*;
// Re-export the added-vocabulary subsystem (AddedToken/AddedVocabulary + vocab_store/buckets modules) at the crate root
pub use added_vocabulary::*;

// Re-export also parallelism utils
pub use utils::parallelism;

// Re-export ProgressFormat for trainer configuration
pub use utils::ProgressFormat;

// Re-export for from_pretrained
#[cfg(feature = "http")]
pub use utils::from_pretrained::FromPretrainedParameters;
