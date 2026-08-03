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
    l: u32, // the left entry
    r: u32, // the right entry
}

const DEAD_RANK: u32 = u32::MAX;
const NONE: u32 = u32::MAX;

struct MergeScratch {
    entries: Vec<Entry>,
    cold: Vec<u32>,
    hot: Vec<u32>,
}

pub fn two_tier_queue_merge(
    tables: BpeTables,
    to_merge: &mut Vec<u32>,
    mut global_min: u64,
    merge_scratch: MergeScratch,
) {
    let sorted_cold = to_merge.sort_unstable();
    todo!()
}
