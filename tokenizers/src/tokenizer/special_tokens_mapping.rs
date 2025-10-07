use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
// A struct that represents the mapping between standard special token names like
// `eos_token` or `bos_token` or `my_token` to the corresponding string tokens.
//
// We choose BTreeMap and set for ordered serialization + fast element check
// Supports updating one entry, the whole entry
// Example
pub struct SpecialTokensMapping {
    inner: BTreeMap<String, BTreeSet<String>>,
}

impl SpecialTokensMapping {
    pub fn new(inner: BTreeMap<String, BTreeSet<String>>) -> Self {
        Self { inner }
    }
}
