#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(clippy::all)]
#![allow(clippy::upper_case_acronyms)]
#![doc(html_favicon_url = "https://huggingface.co/favicon.ico")]
#![doc(html_logo_url = "https://huggingface.co/landing/assets/huggingface_logo.svg")]

//! The reader for the canonical `tokenizer.json` format.
//!
//! Two files and no serde: [`json`] parses with `hifijson` and reassembles floats the way
//! `serde_json` does, and [`from_json`] walks the tree it produces to build
//! [`tk_encode::pipeline::PipelineTokenizer`] directly. Nothing here
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
//! The format is drawn in `SPEC.md`, next to this crate's `Cargo.toml`. This reader accepts it as
//! it is written *today* and refuses anything older with a message naming what to convert:
//!
//! - a `model` with no `"type"`,
//! - `merges` spelled `"a b"` rather than `["a", "b"]`,
//! - a `Metaspace` spelled with `add_prefix_space`,
//! - a vocabulary given as a file path (`files`) — refused here and converted nowhere.
//!
//! The first three are [`tk_convert`](https://docs.rs/tokenizers)'s job:
//! `tk_convert::canonicalize_str` rewrites an old `tokenizer.json` into a canonical one, which is
//! then read here. Keeping the backwards compatibility in a JSON→JSON pass upstream is what lets
//! this crate stay serde-free.
//!
//! ## Features
//!
//! - **`deserialize`** (default) — [`from_json`] / [`from_json_file`], i.e. the whole point of the
//!   crate. An inference build wants exactly this and nothing else.
//! - **`serialize`** (off by default) — [`to_json`] / [`to_json_file`], writing a
//!   `PipelineTokenizer` back out as a canonical `tokenizer.json`. Off by default because an
//!   inference build never writes a config, and because the writer is the half that has to keep
//!   things alive in order to describe them: the `Precompiled` charsmap is in the pipeline for its
//!   benefit alone.
//!
//! ## Writing is not the inverse of reading
//!
//! The reader lowers a config into runtime state, and lowering discards whatever the runtime does
//! not read. So [`to_json`] reconstructs rather than transcribes — a BPE merge list comes back out
//! of the merge tables, a `Metaspace` pre-tokenizer out of the normalizer it became — and what it
//! guarantees is that the file it writes *encodes identically*, not that it is the file that was
//! read. [`to_json`]'s own documentation lists every place the two differ.
#![doc = include_str!("../SPEC.md")]

pub mod json;
mod vendored;

// The reader proper, behind the `deserialize` feature. `json` stays unconditional: the accessors
// are useful on their own and the `serialize` side wants them too.
#[cfg(feature = "deserialize")]
mod from_json;

#[cfg(feature = "deserialize")]
pub use from_json::{from_json, from_json_file};

// The writer. Compiled whenever `serialize` is on, and also in *any* test build — the `test` arm is
// what makes `cargo test --workspace` run the writer's tests, which is where the round-trip gate
// lives. Without it the gate would only ever run under an explicit `--features serialize`, i.e.
// never on CI, and a gate that does not run is not a gate. Nothing escapes into a real build: the
// public functions below stay behind the feature, so an inference build has no writer either way.
#[cfg(any(feature = "serialize", all(test, feature = "deserialize")))]
mod to_json;

#[cfg(feature = "serialize")]
pub use to_json::{to_json, to_json_file};
