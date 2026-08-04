use crate::models::bpe::bpe_build_tables::{BpeTables, ID_MASK, RANK_MASK};
const GATE_MULTI: u16 = 8;
const GATE_ASCII: u16 = 24;

pub fn build_byte_to_gate() -> [u16; 256] {
    let mut b2g = [GATE_MULTI; 256];
    b2g[..0x80].fill(GATE_ASCII);
    // A ByteLevel pre-tokenizer hands us the leading space (" word"), so the first byte says
    // nothing about the script of the rest: " <greek>" would read as ASCII and take the long gate.
    for ws in *b" \t\n\r" {
        b2g[ws as usize] = GATE_MULTI;
    }
    b2g
}

#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct Entry {
    pub rank: u32, // the rank of the merge? but this should be the internal ID.
    pub prod: u32, // the internal ID of the merge (unique as its a product and not a merge)
    pub a: u32,    // the merge is (a,b) these are the internal ids of them
    pub b: u32,
    pub l: u32, // index of the left entry in the cold table
    pub r: u32, // index of the rigthh entry
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
pub struct MergeScratch {
    pub(crate) entries: Vec<Entry>,
    pub cold: Vec<u64>, // even though the values stored can be u32, this makes it simpler to pack the
    // rank and the entry index
    pub hot: Vec<u64>,
}

pub fn two_tier_queue_merge(
    tables: &BpeTables,
    to_merge: &mut Vec<u32>,
    merge_scratch: &mut MergeScratch,
) {
    let MergeScratch { entries, cold, hot } = merge_scratch;
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

    to_merge.clear();
    if head == NONE {
        to_merge.push(single);
        return;
    }
    let mut index = head as usize;
    to_merge.push(entries[index].a);
    loop {
        to_merge.push(entries[index].b);
        match entries[index].r {
            NONE => break,
            next => index = next as usize,
        }
    }
}
