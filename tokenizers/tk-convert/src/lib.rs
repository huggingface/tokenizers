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
//! `["a", "b"]`, a `Metaspace` spelled with `add_prefix_space`, a `Unigram` identified only by the
//! shape of its vocab: every one of those is recognised here and rewritten into the one spelling
//! the slim reader accepts. Anything genuinely ambiguous is *refused* with a [`ConvertError`]
//! naming what a human has to decide, rather than guessed at.
//!
pub mod convert;

pub use convert::{
    ConvertError, canonicalize_file, canonicalize_file_compact, canonicalize_str,
    canonicalize_str_compact, canonicalize_value,
};
