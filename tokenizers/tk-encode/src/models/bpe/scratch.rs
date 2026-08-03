//! Per-thread scratch for BPE. Every buffer here is cleared, never reallocated, so tokenizing a
//! sequence does not allocate.
use crate::models::bpe::two_tier_merge::MergeScratch;
use crate::models::bpe::word_cache::WordCache;
use crate::models::bpe::{Merge, Word};
use crate::pipeline::ModelScratch;
use dary_heap::QuaternaryHeap;

#[derive(Default)]
pub struct BpeScratch {
    /// Symbols of the word being merged. Reused across words so tokenizing allocates nothing.
    pub(crate) to_merge: Vec<u32>,
    /// Entry arena and the two queue tiers, likewise reused.
    pub(crate) merge: MergeScratch,
    pub(crate) merge_queue: QuaternaryHeap<Merge>,
    pub(crate) skip: Vec<Merge>,
    pub(crate) word: Word,
    pub(crate) word_cache: Option<WordCache>,
}

impl ModelScratch for BpeScratch {
    fn clear(&mut self) {
        let Self {
            to_merge,
            merge,
            merge_queue,
            skip,
            word,
            word_cache: _,
        } = self;
        // `clear` keeps each buffer's capacity, which is what makes tokenizing allocation-free
        to_merge.clear();
        merge.entries.clear();
        merge.cold.clear();
        merge.hot.clear();
        merge_queue.clear();
        skip.clear();
        word.clear();
        // The word cache is intentionally kept across clears so it stays warm for future callers
    }
}