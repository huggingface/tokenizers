//! [Unigram](https://arxiv.org/abs/1804.10959) model.
pub mod lattice;
pub mod model;
mod trie;

#[cfg(test)]
mod tests;

pub use lattice::*;
pub use model::*;
