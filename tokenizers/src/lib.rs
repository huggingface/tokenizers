#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_favicon_url = "https://huggingface.co/favicon.ico")]
#![doc(html_logo_url = "https://huggingface.co/landing/assets/huggingface_logo.svg")]

//! The 🤗 Tokenizers library.
//!
//! The implementation is split across crates (each built on internal engines — `tk_encode` on the
//! `bitcannon` SIMD pre-tokenizer, and the shared `bitmap_gen` tables):
//!
//! - [`tk_encode`] — inference: the model engines and the full pipeline components
//!   ([`Normalizer`], [`PreTokenizer`], [`Model`], [`PostProcessor`], [`Decoder`]).
//! - `tk_serialize` — the reader: `from_json_file` turns a canonical `tokenizer.json` into a
//!   [`pipeline::PipelineTokenizer`], with no serde anywhere.
//! - [`tk_convert`] — the upgrade pass: [`canonicalize_file`] rewrites a `tokenizer.json` written by
//!   an older version into the canonical form that reader accepts.
//! - `tk_train` — Types and traits to train tokenizers (gated by the `train` feature).
//!
//! This `tokenizers` crate is a thin umbrella that re-exports them so existing `tokenizers::…`
//! paths keep working.
//!
//! ## Using the crate
//!
//! Add it to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! tokenizers = "1.0.0"
//! ```
//!
//! The default features cover inference: every model, normalizer and pre-tokenizer, batch encoding
//! across threads, and writing a tokenizer back out as JSON.
//!
//! Training tokenizers and downloading from the Hugging Face Hub are opt-in:
//!
//! ```toml
//! tokenizers = { version = "1.0.0", features = ["train", "http"] }
//! ```
//!
//! For a smaller build, turn the defaults off and keep only what your `tokenizer.json` needs.
//! A config that names a component you compiled out fails at load.
//!
//! ```toml
//! tokenizers = { version = "1.0.0", default-features = false, features = ["bpe"] }
//! ```
//!
//! | Feature | Default | What it turns on |
//! |---|---|---|
//! | `bpe` | yes | The BPE model. With `train`, the BPE trainer too. |
//! | `unigram` | yes | The Unigram model. With `train`, the Unigram trainer too. |
//! | `wordpiece` | yes | The WordPiece model. With `train`, the WordPiece trainer too. That trainer runs the BPE one, so it also turns on `bpe`. |
//! | `wordlevel` | yes | The WordLevel model. With `train`, the WordLevel trainer too. |
//! | `normalizers` | yes | The normalizers that need Unicode tables: `NFC`, `NFD`, `NFKC`, `NFKD`, `Nmt`, `BertNormalizer`, `Precompiled` and `StripAccents`. The others are always built. |
//! | `unicode-scripts` | yes | The `UnicodeScripts` pre-tokenizer. |
//! | `parallelism` | yes | Batch encoding across threads, with rayon. |
//! | `serialize` | yes | Writing a tokenizer back out as `tokenizer.json` (`to_json`). Reading is always built. |
//! | `progressbar` | yes | Progress bars during training. Does nothing without `train`. |
//! | `train` | no | The trainers: `Trainer`, `TrainerWrapper` and the `trainers` module. |
//! | `esaxx_fast` | no | The C++ suffix array for Unigram training, instead of the pure-Rust one. Turns on `train` and `unigram`. |
//! | `parity-aware-bpe` | no | `ParityBpeTrainer`, which trains one BPE vocabulary over several corpora, one per language. Turns on `train` and `bpe`. |
//! | `http` | no | `utils::from_pretrained`, which downloads a `tokenizer.json` from the Hugging Face Hub. |
//! | `regex` | no | A regex engine (`fancy-regex`) for `Split` and `Replace` regex patterns that the built-in pre-tokenizer does not cover. Without it, those configs fail at load. Plain-string patterns work either way. |
//!
//! ## What rc0 does not have
//!
//! The `Tokenizer` object model — `Tokenizer::new`, the component setters, `add_tokens`,
//! `save`, `from_pretrained`, truncation and padding — is **not** in this release.
//! [`pipeline::PipelineTokenizer`] is read-only: it encodes and decodes what a `tokenizer.json`
//! describes and has no way to be built up or written back out. See
//! `REQUIRED_FOR_V1.md` at the repository root for the full list and why each one is deferred.
//!
//! ## Tokenization example
//!
//! Read a config and encode with it. The example lives in `tk-serialize`, which is the only crate
//! that can compile it.

// ---------------------------------------------------------------------------
// Inference — re-exported from `tk-encode`.
// ---------------------------------------------------------------------------
pub use tk_encode::{
    decoders, models, normalizers, pipeline, pre_tokenizers, processors, tokenizer, utils, vocab,
};

// Mirror the v1 top-level re-exports (`pub use tokenizer::*;` etc.).
pub use tk_encode::tokenizer::*;
#[cfg(feature = "parallelism")]
pub use tk_encode::utils::parallelism;

#[cfg(feature = "http")]
pub use tk_encode::FromPretrainedParameters;

// ---------------------------------------------------------------------------
// The reader and the writer — re-exported from `tk-serialize`.
// ---------------------------------------------------------------------------
pub use tk_serialize::{from_json, from_json_file, json, post_processor_from_json};
#[cfg(feature = "serialize")]
pub use tk_serialize::{post_processor_to_json, str_to_json, to_json};

// ---------------------------------------------------------------------------
// The legacy-config upgrade pass — all that is left of the config layer.
// ---------------------------------------------------------------------------
pub use tk_convert::{
    ConvertError, canonicalize_file, canonicalize_post_processor, canonicalize_str,
    canonicalize_value, convert,
};

// ---------------------------------------------------------------------------
// Training — re-exported from `tk-train`, behind the `train` feature.
// ---------------------------------------------------------------------------
#[cfg(feature = "train")]
pub use tk_train::{ModelWrapper, ProgressFormat, Trainable, Trainer, TrainerWrapper, trainers};
