#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(clippy::all)]
#![doc(html_favicon_url = "https://huggingface.co/favicon.ico")]
#![doc(html_logo_url = "https://huggingface.co/landing/assets/huggingface_logo.svg")]

//! The legacy `tokenizer.json` upgrade pass.
//!
//! One job: read a `tokenizer.json` written by any older version of this library and emit the
//! *canonical* form of the same config, so that `tk_serialize`'s canonical-only reader can read it.
//! It is a JSON→JSON rewrite — [`canonicalize_str`], [`canonicalize_file`],
//! [`canonicalize_value`] — and it depends on nothing but `std::path` and `serde_json`.
//!
//! Configs written before the `"type"` tag existed, merges spelled `"a b"` rather than
//! `["a", "b"]`, a `Metaspace` spelled with `add_prefix_space`, a vocabulary given as a file path,
//! a `Unigram` identified only by the shape of its vocab: every one of those is recognised here and
//! rewritten into the one spelling the slim reader accepts. Anything genuinely ambiguous is
//! *refused* with a [`ConvertError`] naming what a human has to decide, rather than guessed at.
//!
//! ## What this crate used to be
//!
//! Until rc0 it was the whole config layer: `Tokenizer`, `TokenizerImpl`, `TokenizerBuilder`, the
//! five component wrapper enums and their hand-written serde, the config-shaped models,
//! `AddedVocabulary`, and the lowering from a config into `tk-encode`'s runtime pipeline — about
//! 7,500 lines. All of it is gone, and with it this crate's dependency on `tk-encode` and on serde
//! itself. The runtime is `tk_encode::pipeline::PipelineTokenizer`; the reader that builds one from
//! a canonical file is `tk-serialize`; this crate is only the bridge from *old* files to that
//! reader.
//!
//! `REQUIRED_FOR_V1.md` at the repository root records what that deletion dropped and what v1 has
//! to bring back.

/// The JSON→JSON pass that fills in what an old `tokenizer.json` leaves out, so that
/// [`tk_serialize`](https://docs.rs/tokenizers)'s canonical-only reader can read it.
pub mod convert;

pub use convert::{ConvertError, canonicalize_file, canonicalize_str, canonicalize_value};
