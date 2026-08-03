use itertools::Merge;

use crate::models::bpe::tables::BpeTables;
const GATE_MULTI: u16 = 8;
const GATE_ASCII: u16 = 24;

pub fn build_byte_to_gate() -> [u16; 256] {
    let mut b2g = [0u16; 256];
    for b in 0..256 {
        if b < 0x80 {
            b2g[b] = GATE_ASCII;
        } else {
            b2g[b] = GATE_MULTI;
        }
    }
    b2g
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Entry {
    rank: u32, // the rank of the merge? but this should be the internal ID.
    prod: u32, // the internal ID of the merge (unique as its a product and not a merge)
    a: u32,    // the merge is (a,b) these are the internal ids of them
    b: u32,
    l: u32, // index of the left entry in the cold table
    r: u32, // index of the rigthh entry
}

const DEAD_RANK: u32 = u32::MAX;
const NONE: u32 = u32::MAX;

#[derive(Default)]
pub struct MergeScratch {
    pub entries: Vec<Entry>,
    pub cold: Vec<u64>, // even though the values stored can be u32, this makes it simpler to pack the
    // rank and the entry index
    pub hot: Vec<u64>,
}

pub fn two_tier_queue_merge(
    tables: &BpeTables,
    to_merge: &mut Vec<u32>,
    merge_scratch: &mut MergeScratch,
) {
    for id in 0..to_merge.len() - 1 {
        // we need to create the entries
        let rank = to_merge[id];
        let next = to_merge[id + 1];
        merge_scratch.entries.push(Entry {
            rank: rank,
            prod: (tables.get_value(&rank, &next) >> 32) as u32,
            a: tables,
        })
    }
    merge_scratch.cold = to_merge
        .iter()
        .enumerate()
        .map(|(i, n)| (*n as u64) << 32 | i as u64)
        .collect();
    todo!()
}
