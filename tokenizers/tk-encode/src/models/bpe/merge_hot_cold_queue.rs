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
//! `l`/`r`. A merge rewrites its neighbour entries in place and pushes their new keys to hot; the
//! keys already queued for those neighbours go stale, and are recognized and skipped because
//! their rank half no longer matches the entry's current rank. When the queue runs dry, the
//! merged word is read back by walking the links from the leftmost live entry.
use crate::models::bpe::tables::{BpeTables, ID_MASK, RANK_MASK};

#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct Entry {
    /// Rank of the merge (a, b), `DEAD_RANK` when the pair does not merge or was consumed.
    pub rank: u32,
    /// Internal id of the token (a, b) merges into.
    pub prod: u32,
    /// The pair's symbols, as internal ids.
    pub a: u32,
    pub b: u32,
    /// Arena index of the pair to the left, `NONE` at the word's start.
    pub l: u32,
    /// Arena index of the pair to the right, `NONE` at the word's end.
    pub r: u32,
}

const DEAD_RANK: u32 = u32::MAX;
const NONE: u32 = u32::MAX;
const NO_MERGE: u64 = u64::MAX;
const EMPTY_KEY: u64 = u64::MAX;

impl Entry {
    pub fn update(self, tables: &BpeTables, entries: &mut [Entry], hot: &mut Vec<u64>) {
        if self.l != NONE {
            // left pair becomes (ent[l].a, prod)
            let left = &mut entries[self.l as usize];
            let key = tables.get_value(&left.a, &self.prod);
            left.b = self.prod;
            left.rank = (key >> 32) as u32;
            left.prod = (key & ID_MASK) as u32;
            left.r = self.r;
            if key != NO_MERGE {
                hot_push(hot, (key & RANK_MASK) | self.l as u64)
            }
        }

        if self.r != NONE {
            // right pair becomes (prod, ent[r].b)
            let right = &mut entries[self.r as usize];
            let key = tables.get_value(&self.prod, &right.b);
            right.a = self.prod;
            right.rank = (key >> 32) as u32;
            right.prod = (key & ID_MASK) as u32;
            right.l = self.l;
            if key != NO_MERGE {
                hot_push(hot, (key & RANK_MASK) | self.r as u64)
            }
        }
    }
}

#[inline(always)]
fn hot_push(hot: &mut Vec<u64>, key: u64) {
    hot.push(key);
    let mut child = hot.len() - 1;
    while child > 0 {
        let parent = (child - 1) / 2;
        if hot[parent] <= key {
            break;
        }
        hot[child] = hot[parent];
        child = parent;
    }
    hot[child] = key;
}

#[inline(always)]
fn hot_pop(hot: &mut Vec<u64>) -> u64 {
    let top = hot[0];
    let last = hot.pop().unwrap();
    let len = hot.len();
    if len == 0 {
        return top;
    }
    let mut parent = 0usize;
    loop {
        let left = 2 * parent + 1;
        if left >= len {
            break;
        }
        let right = left + 1;
        let child = if right < len && hot[right] < hot[left] {
            right
        } else {
            left
        };
        if hot[child] >= last {
            break;
        }
        hot[parent] = hot[child];
        parent = child;
    }
    hot[parent] = last;
    top
}

#[derive(Default)]
pub struct QueueScratch {
    /// One [`Entry`] per adjacent pair of the word, linked left/right.
    pub(crate) entries: Vec<Entry>,
    /// The initial pairs' keys, sorted once; see the module docs.
    pub cold: Vec<u64>,
    /// The merge-created pairs' keys, kept as a binary min-heap.
    pub hot: Vec<u64>,
}

/// Merges the word whose pair entries and cold keys `scratch` holds (filled by `convert_queue`),
/// writing the merged word into `symbols` as internal ids.
pub fn merge_hot_cold_queue(
    tables: &BpeTables,
    symbols: &mut Vec<u32>,
    scratch: &mut QueueScratch,
) {
    let QueueScratch { entries, cold, hot } = scratch;
    if entries.is_empty() {
        return;
    }
    // sort the cold only once.
    cold.sort_unstable();
    hot.clear();
    let (mut head, mut single) = (0u32, 0u32);
    let mut cursor = 0usize;

    loop {
        let cold_key = cold.get(cursor).copied().unwrap_or(EMPTY_KEY);
        let hot_key = hot.first().copied().unwrap_or(EMPTY_KEY);
        let key = if cold_key <= hot_key {
            if cold_key == EMPTY_KEY {
                break;
            }
            cursor += 1;
            cold_key
        } else {
            hot_pop(hot)
        };
        let index = key as u32 as usize;
        let entry = entries[index];
        if entry.rank as u64 != key >> 32 {
            continue;
        }
        entries[index].rank = DEAD_RANK;
        if entry.l == NONE {
            head = entry.r;
            single = entry.prod // pretoken collapsed to one token
        }
        entry.update(tables, entries, hot);
    }

    symbols.clear();
    if head == NONE {
        symbols.push(single);
        return;
    }
    let mut index = head as usize;
    symbols.push(entries[index].a);
    loop {
        symbols.push(entries[index].b);
        match entries[index].r {
            NONE => break,
            next => index = next as usize,
        }
    }
}
