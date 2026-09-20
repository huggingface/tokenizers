//! The hot/cold queue: the merge engine for longer words.
//!
//! Multipass (see `merge_multipass`) re-sweeps the whole word once per target merge, which gets
//! expensive as words grow. This engine instead applies pairs in rank order straight away, from a
//! priority queue split in two tiers:
//!
//! - **cold**: the pairs the word starts with. They are all known before the first merge, so a
//!   plain vector sorted once and consumed front to back beats a heap.
//! - **hot**: the pairs each merge creates with its new neighbours. Only these need a live
//!   priority queue, a small binary min-heap.
//!
//! Each step takes the lower of the next cold key and the top hot key. A key packs
//! `rank << 32 | entry index`, so comparing keys compares ranks first; entries are created in
//! word order and updated in place, so on equal ranks the smaller index is the leftmost pair,
//! which is the order BPE prescribes.
//!
//! The word itself is the `entries` arena: one [`Entry`] per adjacent pair, doubly linked through
//! `left_pair_index`/`right_pair_index`. A merge rewrites its neighbour entries in place and pushes
//! their new keys to hot; the keys already queued for those neighbours go stale, and are recognized
//! and skipped because their rank half no longer matches the entry's current rank. When the queue
//! runs dry, the merged word is read back by walking the links from the leftmost live entry.
use crate::models::bpe::tables::{BpeTables, ID_MASK, RANK_MASK};

/// Rank of an [`Entry`] that will not be merged, either because its pair has no merge at all or
/// because its merge was applied already.
const DEAD_RANK: u32 = u32::MAX;
/// No entry, which is what the links of the pairs at both ends of the word hold.
const NONE: u32 = u32::MAX;
/// What [`BpeTables::get_value`] returns for a pair that does not merge.
const NO_MERGE: u64 = u64::MAX;
/// Stands in for the next key of a drained tier: above every real key, so the other tier wins.
const EMPTY_KEY: u64 = u64::MAX;

/// Merges the word whose pair entries and queue keys `scratch` holds (filled by `convert_queue`),
/// writing the merged word into `symbols` as internal ids.
pub fn merge_with_queue(tables: &BpeTables, symbols: &mut Vec<u32>, scratch: &mut QueueScratch) {
    let QueueScratch { entries, queue } = scratch;
    if entries.is_empty() {
        return;
    }
    queue.sort_cold();

    // Entry 0 is the leftmost pair, and only merging the leftmost pair moves it.
    let mut leftmost = 0u32;
    // The whole word as one symbol, read back only when `leftmost` ends up `NONE`, which takes the
    // leftmost pair to have been merged.
    let mut collapsed_symbol = 0u32;

    while let Some(key) = queue.pop() {
        let index = key as u32 as usize;
        let entry = entries[index];
        // The entry was rewritten after this key was queued, so the key is stale.
        if entry.rank as u64 != key >> 32 {
            continue;
        }
        entries[index].rank = DEAD_RANK;
        if entry.left_pair_index == NONE {
            leftmost = entry.right_pair_index;
            collapsed_symbol = entry.merged_symbol;
        }
        entry.rewrite_neighbours(tables, entries, queue);
    }

    symbols.clear();
    if leftmost == NONE {
        symbols.push(collapsed_symbol);
        return;
    }
    let mut index = leftmost as usize;
    symbols.push(entries[index].left_symbol);
    loop {
        symbols.push(entries[index].right_symbol);
        match entries[index].right_pair_index {
            NONE => break,
            next => index = next as usize,
        }
    }
}

/// The buffers one word's merge needs, reused from word to word so that merging allocates nothing.
#[derive(Default)]
pub struct QueueScratch {
    /// One [`Entry`] per adjacent pair of the word, linked left/right.
    pub(crate) entries: Vec<Entry>,
    pub(crate) queue: MergeQueue,
}

/// The two-tier priority queue over pair keys: a sorted vector for the pairs the word starts with,
/// a binary min-heap for the pairs merges create. The module docs say why.
#[derive(Default)]
pub(crate) struct MergeQueue {
    /// Keys of the pairs the word starts with, in the order [`MergeQueue::sort_cold`] put them.
    cold: Vec<u64>,
    /// Keys of the pairs merges create, as a binary min-heap.
    hot: Vec<u64>,
    /// How far into `cold` [`MergeQueue::pop`] has walked.
    cold_cursor: usize,
}

impl MergeQueue {
    /// Empties both tiers for a new word, leaving room for `pairs` cold keys.
    pub(super) fn clear(&mut self, pairs: usize) {
        self.cold.clear();
        self.cold.reserve(pairs);
        self.hot.clear();
        self.cold_cursor = 0;
    }

    /// Queues one of the pairs the word starts with. Every cold key is pushed before the first
    /// [`MergeQueue::pop`], which is what lets one sort replace a heap.
    #[inline(always)]
    pub(super) fn push_cold(&mut self, key: u64) {
        self.cold.push(key);
    }

    /// Queues a pair a merge created.
    #[inline(always)]
    fn push(&mut self, key: u64) {
        heappush(&mut self.hot, key);
    }

    /// Sorts the cold tier, so that [`MergeQueue::pop`] can read it front to back. Called once,
    /// after the conversion has pushed every cold key.
    fn sort_cold(&mut self) {
        self.cold.sort_unstable();
    }

    /// The lowest key of the two tiers, `None` once both are drained.
    #[inline(always)]
    fn pop(&mut self) -> Option<u64> {
        let cold_key = self
            .cold
            .get(self.cold_cursor)
            .copied()
            .unwrap_or(EMPTY_KEY);
        let hot_key = self.hot.first().copied().unwrap_or(EMPTY_KEY);
        if cold_key <= hot_key {
            if cold_key == EMPTY_KEY {
                return None;
            }
            self.cold_cursor += 1;
            Some(cold_key)
        } else {
            Some(heappop(&mut self.hot))
        }
    }
}

// Binary heap vendored from `std::collections::BinaryHeap`, in min-order over a raw u64 key (rank
// in the high bits, entry index in the low): std is a max-heap, and wrapping in Reverse would cost
// us the packed key. Same algorithm as std -- sift-up on push, sift-down with early exit on pop
// (`sift_up` / `sift_down_range`) -- and the same "hole" form: the parent or child is shifted into
// the hole and the key written once at the end, instead of a swap per level.
#[inline(always)]
fn heappush(heap: &mut Vec<u64>, key: u64) {
    heap.push(key);
    let mut hole = heap.len() - 1;
    while hole > 0 {
        let parent = (hole - 1) / 2;
        if heap[parent] <= key {
            break;
        }
        heap[hole] = heap[parent];
        hole = parent;
    }
    heap[hole] = key;
}

/// Removes and returns the smallest key. The heap must not be empty.
#[inline(always)]
fn heappop(heap: &mut Vec<u64>) -> u64 {
    let top = heap[0];
    let tail = heap.pop().unwrap();
    let len = heap.len();
    if len == 0 {
        return top;
    }
    let mut hole = 0usize;
    loop {
        let left_child = 2 * hole + 1;
        if left_child >= len {
            break;
        }
        let right_child = left_child + 1;
        let child = if right_child < len && heap[right_child] < heap[left_child] {
            right_child
        } else {
            left_child
        };
        if heap[child] >= tail {
            break;
        }
        heap[hole] = heap[child];
        hole = child;
    }
    heap[hole] = tail;
    top
}

/// One adjacent pair of the word: the two symbols, the merge the tables have for them, and the
/// pairs on either side of it.
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct Entry {
    /// Rank (priority) of the merge, `DEAD_RANK` when the pair does not merge or was consumed.
    pub rank: u32,
    /// The pair's symbols, as internal ids.
    pub left_symbol: u32,
    pub right_symbol: u32,
    /// The symbol (internal id) of the merged pair. Only meaningful while `rank` is not
    /// `DEAD_RANK`: a pair that does not merge has no merged symbol.
    pub merged_symbol: u32,
    /// Arena index of the pair to the left, `NONE` at the word's start.
    pub left_pair_index: u32,
    /// Arena index of the pair to the right, `NONE` at the word's end.
    pub right_pair_index: u32,
}

impl Entry {
    /// Applies this entry's merge to the pairs on either side of it: each takes `merged_symbol` for
    /// the symbol it shared with this pair, links past this entry, and queues the key of the pair
    /// it has become.
    fn rewrite_neighbours(self, tables: &BpeTables, entries: &mut [Entry], queue: &mut MergeQueue) {
        if self.left_pair_index != NONE {
            // the pair to the left becomes (its own left symbol, merged_symbol)
            let left_pair = &mut entries[self.left_pair_index as usize];
            let merge = tables.get_value(&left_pair.left_symbol, &self.merged_symbol);
            left_pair.right_symbol = self.merged_symbol;
            left_pair.rank = (merge >> 32) as u32;
            left_pair.merged_symbol = (merge & ID_MASK) as u32;
            left_pair.right_pair_index = self.right_pair_index;
            if merge != NO_MERGE {
                queue.push((merge & RANK_MASK) | self.left_pair_index as u64)
            }
        }

        if self.right_pair_index != NONE {
            // the pair to the right becomes (merged_symbol, its own right symbol)
            let right_pair = &mut entries[self.right_pair_index as usize];
            let merge = tables.get_value(&self.merged_symbol, &right_pair.right_symbol);
            right_pair.left_symbol = self.merged_symbol;
            right_pair.rank = (merge >> 32) as u32;
            right_pair.merged_symbol = (merge & ID_MASK) as u32;
            right_pair.left_pair_index = self.left_pair_index;
            if merge != NO_MERGE {
                queue.push((merge & RANK_MASK) | self.right_pair_index as u64)
            }
        }
    }
}
