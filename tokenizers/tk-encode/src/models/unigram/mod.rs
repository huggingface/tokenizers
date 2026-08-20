//! [Unigram](https://arxiv.org/abs/1804.10959) model.
pub mod lattice;
pub mod model;
#[cfg(feature = "serde")]
mod serialization;
mod trie;

pub use lattice::*;
pub use model::*;
