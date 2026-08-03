//! Per-thread scratch for BPE. Every buffer here is cleared, never reallocated, so tokenizing a
//! sequence does not allocate.
use crate::models::bpe::merge_hot_cold_queue::MergeScratch;
use crate::models::bpe::{Merge, Word};
use crate::pipeline::ModelScratch;
use dary_heap::QuaternaryHeap;

pub struct BpeScratch {
    /// Symbols of the word being merged. Reused across words so tokenizing allocates nothing.
    pub(crate) to_merge: Vec<u32>,
    /// Entry arena and the two queue tiers, likewise reused.
    pub(crate) merge: MergeScratch,
    pub(crate) merge_queue: QuaternaryHeap<Merge>,
    pub(crate) skip: Vec<Merge>,
    pub(crate) word: Word,
}

impl ModelScratch for BpeScratch {}
