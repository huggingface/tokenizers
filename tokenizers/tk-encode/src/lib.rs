#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(clippy::all)]
#![allow(clippy::upper_case_acronyms)]
#![doc(html_favicon_url = "https://huggingface.co/favicon.ico")]
#![doc(html_logo_url = "https://huggingface.co/landing/assets/huggingface_logo.svg")]

//! The inference half of `tokenizers`, written in Rust.
//! Provides an implementation of today's most used tokenizers, with a focus on performance and
//! versatility.
//!
//! # What is a Tokenizer
//!
//! A tokenizer works as a pipeline, it processes some raw text as input and outputs an `Encoding`.
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
//! ## Loading a tokenizer and encoding
//!
//! This crate is the runtime only: it knows how to *encode*, not what a `tokenizer.json` looks
//! like. Reading one is `tk-serialize`'s job — `tk_serialize::from_json_file` returns the
//! [`pipeline::PipelineTokenizer`] built here, from a hand-rolled JSON reader with no serde
//! anywhere. (The example lives there, because only that crate can compile it.)
//!
//! The `Tokenizer` orchestration, `from_pretrained`, `save`, and every backwards-compatible config
//! shape live one crate further out again, in `tk-convert` (both re-exported by the `tokenizers`
//! umbrella crate), so an encode-only build links none of it.
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
//!   With this feature enabled, `tk_convert`'s `Tokenizer::from_pretrained` becomes
//!   accessible.
//!
//! - **serde**: Every component's own `Serialize`/`Deserialize` — the derives on the types
//!   themselves and the hand-written impls in the `serialization.rs` next to each of them. Off by
//!   default, because the slim reader (`tk-serialize`) builds a pipeline from a canonical
//!   `tokenizer.json` with no serde at all; `tk-convert` and the `tokenizers` umbrella turn it on,
//!   so python, node and existing `tokenizers::…` users are unaffected.
//!
//! - **bpe** / **unigram** / **wordpiece** / **wordlevel**: one per model. Only `bpe` is on by
//!   default, because none of the current SOTA models use anything else. A `tokenizer.json` naming
//!   a model whose feature is off is refused at load rather than mis-read.
//!
//! - **normalizers**: the table-backed normalizers (NFC/NFD/NFKC/NFKD, `Nmt`, `StripAccents`,
//!   `Bert`, SentencePiece's precompiled charsmap) and the ~150 KB of static Unicode tables behind
//!   them.
//!
//! - **unicode-scripts**: the `UnicodeScripts` pre-tokenizer. Its script table is a 64 KiB
//!   `LazyLock` plus ~19 KB of generated range matching, and no widely-used `tokenizer.json`
//!   declares it — hence off by default.
//!
//! - **parallelism**: rayon-backed batch encoding.
//!
//! - **fancy-regex**: the optional system-regex backend, needed *only* for a genuine regex pattern
//!   in a `Split` pre-tokenizer or a `Replace` normalizer. The atomsplit-native pre-tokenizers
//!   (GPT-2, cl100k, o200k, tekken, deepseek, the class family, char-delimiter) need no backend,
//!   and a literal pattern is searched for directly.
//!
//! # Building small
//!
//! The default build is the slim one: no serde (the `serde` feature is off), and only BPE.
//! `make slim-size` prints its stripped and gzipped size.
//!
//! The largest remaining cost is not in this crate: `std`'s panic, unwinding and
//! backtrace-symbolisation machinery (`addr2line`, `gimli`, `rustc_demangle`, `object`, plus the
//! build machine's source paths baked into `__cstring`) is about 40% of the gzipped binary. A stable
//! toolchain cannot drop it; a nightly one can:
//!
//! ```text
//! cargo +nightly build --profile minsize \
//!   -Z build-std=std,panic_abort \
//!   -Z build-std-features=panic_immediate_abort ...
//! ```
//!
//! Measured, same features and profile: 360,045 -> 213,744 bytes gzipped (-40.6%). The trade is
//! real: every panic becomes an immediate `abort` with no message, unwinding or backtrace, so a
//! panicking input looks like any other crash. Usually fine on-device, rarely fine on a server.
//! Worth pairing with `--remap-path-prefix` — not for size (~175 bytes compressed) but because a
//! shipped binary should not contain a developer's home directory.

#[macro_use]
extern crate log;

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
pub mod vocab;

// Re-export from tokenizer
pub use tokenizer::*;

// Re-export also parallelism utils
#[cfg(feature = "parallelism")]
pub use utils::parallelism;

// Re-export ProgressFormat for trainer configuration
pub use utils::ProgressFormat;

// Re-export for from_pretrained
#[cfg(feature = "http")]
pub use utils::from_pretrained::FromPretrainedParameters;
