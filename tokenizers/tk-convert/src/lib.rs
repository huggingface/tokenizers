#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(clippy::all)]
#![allow(clippy::upper_case_acronyms)]
#![doc(html_favicon_url = "https://huggingface.co/favicon.ico")]
#![doc(html_logo_url = "https://huggingface.co/landing/assets/huggingface_logo.svg")]

//! The `tokenizer.json` config layer of `tokenizers`.
//!
//! [`tk_encode`] is the runtime: the component structs, the model engines, and the
//! `PipelineTokenizer` encode path with its hand-rolled JSON reader. This crate is everything
//! *around* a config file, and nothing that runs per token:
//!
//! - the five **wrapper enums** — [`ModelWrapper`], [`NormalizerWrapper`], [`PreTokenizerWrapper`],
//!   [`PostProcessorWrapper`], [`DecoderWrapper`] — and their hand-written `Deserialize` impls,
//! - the **config-shaped [`BPE`](models::bpe::BPE)**: the builder with the ten options a config can
//!   name, its `vocab.json` + `merges.txt` loaders, its serde, and the lowering that turns one into
//!   the runtime's `PipelineBPE`. You cannot construct a `BPE` and encode with it *as* the runtime
//!   any more — but an old serialized BPE still reads, and this is what reads it,
//! - the four `Sequence` components, each of which is a `Vec` of one of those wrappers,
//! - the [`Tokenizer`] / [`TokenizerImpl`] / [`TokenizerBuilder`] orchestration and its
//!   `from_file` / `from_str` / `from_bytes` / `from_pretrained` / `save` surface,
//! - the legacy [`AddedVocabulary`],
//! - the [`lowering`] from any of that into `tk-encode`'s runtime pipeline.
//!
//! ## Why the split is where it is
//!
//! A wrapper enum's `Deserialize` names every variant, so *mentioning the wrapper* makes every
//! model and normalizer reachable — a match arm is enough. That reachability was most of what an
//! on-device build was paying for, so it now lives on this side of the line and an encode-only build
//! links none of it: `tk-encode`'s `serde` feature is off by default, and this crate is what turns
//! it on.
//!
//! Backwards compatibility lives here too, all of it. Configs written before the `"type"` tag
//! existed, merges spelled `"a b"` rather than `["a", "b"]`, a `Metaspace` spelled with
//! `add_prefix_space`, a vocabulary given as a file path: every one of those shapes is read here,
//! and refused by the slim reader with a message naming what to convert.
//!
//! ## What stayed behind, and why
//!
//! One thing that reads like config work is still in `tk-encode`, for a reason that is about Rust
//! rather than about design:
//!
//! - **`Model::save`**, because it is a trait method that both bindings implement. The *bodies* that
//!   needed serde did move: `WordLevel` and `Unigram` are written by
//!   [`ModelWrapper::save`](models::ModelWrapper) rather than by their own impls, which is as far as
//!   the orphan rule allows — `Model` and those two types are both defined in `tk-encode`, so no
//!   other crate may write `impl Model for WordLevel`. `WordPiece::save` writes a `vocab.txt`, needs
//!   no serde, and stayed whole.
//!
//! What used to be here as well, and no longer is:
//!
//! - **the leaf components' serde**. Every component's `Serialize`/`Deserialize` sits next to the
//!   type in `tk-encode`, behind that crate's off-by-default `serde` feature, which this crate turns
//!   on. `BPE` is the exception only because the *type* moved here.
//! - **`BPE::read_file` and friends**. They were said to have to stay because the bindings call them
//!   as inherent associated functions; the bindings were changed to follow the types instead.
//!   `BPE`'s came along with `BPE` and are still spelled `BPE::read_file(..)`; `WordPiece`'s and
//!   `WordLevel`'s are free functions in [`models::wordpiece`] and [`models::wordlevel`], because
//!   those two models stayed in the runtime and an inherent `impl` has to live with its type.

#[macro_use]
extern crate log;

/// The JSON→JSON pass that fills in what an old `tokenizer.json` leaves out, so that
/// [`tk_serialize`](https://docs.rs/tokenizers)'s canonical-only reader can read it.
pub mod convert;
pub mod decoders;
pub mod lowering;
mod macros;
pub mod models;
pub mod normalizers;
pub mod pre_tokenizers;
pub mod processors;
pub mod tokenizer;

pub use convert::{ConvertError, canonicalize_file, canonicalize_str, canonicalize_value};
pub use decoders::DecoderWrapper;
pub use models::ModelWrapper;
pub use normalizers::NormalizerWrapper;
pub use pre_tokenizers::PreTokenizerWrapper;
pub use processors::PostProcessorWrapper;

// Mirrors `tk-encode`'s `pub use tokenizer::*`, so `tk_convert::Tokenizer` and
// `tk_convert::tokenizer::Tokenizer` both resolve — the umbrella crate re-exports both shapes.
pub use tokenizer::*;
