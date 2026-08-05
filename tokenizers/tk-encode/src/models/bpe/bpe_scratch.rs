//! Per-thread scratch for BPE. Every buffer here is cleared, never reallocated, so tokenizing a
//! sequence does not allocate.
use crate::models::bpe::merge_hot_cold_queue::MergeScratch;
use crate::pipeline::ModelScratch;

pub struct BpeScratch {
    /// Symbols of the word being merged. Reused across words so tokenizing allocates nothing.
    pub(crate) symbols: Vec<u32>,
    /// Entry arena and the two queue tiers, likewise reused.
    pub(crate) merge: MergeScratch,
}

impl ModelScratch for BpeScratch {}
