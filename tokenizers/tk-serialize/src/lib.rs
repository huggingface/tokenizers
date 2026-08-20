#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(clippy::all)]
#![allow(clippy::upper_case_acronyms)]
#![doc(html_favicon_url = "https://huggingface.co/favicon.ico")]
#![doc(html_logo_url = "https://huggingface.co/landing/assets/huggingface_logo.svg")]

//! The reader for the canonical `tokenizer.json` format.
//!
//! Two files and no serde: [`json`] is a hand-rolled JSON parser, and [`from_json`] walks the tree
//! it produces to build [`tk_encode::pipeline::PipelineTokenizer`] directly. Nothing here
//! deserializes *into* a config type, because there are no config types on this path — the reader
//! constructs the runtime components straight away.
//!
//! ```no_run
//! let tokenizer = tk_serialize::from_json_file("./tokenizer.json")?;
//! let ids = tokenizer.encode("Hey there!", false).wait()?;
//! # Ok::<(), tk_encode::Error>(())
//! ```
//!
//! ## Why this is a crate and not a module
//!
//! Naming a wrapper enum makes every one of its variants reachable, and that reachability was most
//! of what an on-device build used to pay for. `tk-encode` is now the runtime alone; this crate is
//! the only thing that knows what a `tokenizer.json` looks like; and the legacy shapes live one
//! crate further out again, in `tk-convert`.
//!
//! ## Canonical only
//!
//! This reader accepts the format as it is written *today* and refuses anything older with a
//! message naming what to convert:
//!
//! - a `model` with no `"type"`,
//! - `merges` spelled `"a b"` rather than `["a", "b"]`,
//! - a `Metaspace` spelled with `add_prefix_space`,
//! - a vocabulary given as a file path.
//!
//! Every one of those is [`tk_convert`](https://docs.rs/tokenizers)'s job: `tk_convert::canonicalize_str`
//! rewrites an old `tokenizer.json` into a canonical one, which is then read here. Keeping the
//! backwards compatibility in a JSON→JSON pass upstream is what lets this crate stay serde-free.
//!
//! ## Features
//!
//! - **`deserialize`** (default) — [`from_json`] / [`from_json_file`], i.e. the whole point of the
//!   crate. An inference build wants exactly this and nothing else.
//! - **`serialize`** (off by default) — reserved for writing a `PipelineTokenizer` back out as a
//!   canonical `tokenizer.json`. Nothing is gated on it yet: the authoring surface lives in
//!   `tk-convert`, which already serializes a `Tokenizer`. The name is declared here so that an
//!   inference build never has to opt *out* of a writer it does not want.

pub mod json;

// The reader proper, behind the `deserialize` feature. `json` stays unconditional: the hand-rolled
// parser is useful on its own and the `serialize` side would want it too.
#[cfg(feature = "deserialize")]
mod from_json;

#[cfg(feature = "deserialize")]
pub use from_json::{from_json, from_json_file};
