use ptr_hash::{bucket_fn::Linear, PtrHash, PtrHashParams};

use crate::models::bpe::{
    pair_packing::{make_packed_merge, PackedMerge, KEY_MASK},
    word::MergeLookup,
    MergeMap,
};

pub struct MergeStore {
    /// Perfect Hash mapping a packed pair (u64) to index in the slot buffer
    mphf: PtrHash<PackedMerge, Linear>,
    /// Contiguous buffer of [`PackedMerge`] - merge pair + rank of the merge
    slots: Box<[PackedMerge]>,
    /// Contiguous buffer indexed by the same value as slots, holding the merged token id (u32)
    new_ids: Box<[u32]>,
}

impl MergeStore {
    pub fn new() -> Self {
        let empty: [PackedMerge; 0] = [];
        Self {
            mphf: PtrHash::<PackedMerge, Linear>::new(&empty, PtrHashParams::default_fast()),
            slots: Box::new([]),
            new_ids: Box::new([]),
        }
    }

    pub fn build(merges: &MergeMap) -> Self {
        let len = merges.len();

        let (keys, values, ids): (Vec<PackedMerge>, Vec<PackedMerge>, Vec<u32>) = merges
            .iter()
            .map(|(&(left, right), &(rank, new_id))| {
                let packed_merge = make_packed_merge(left, right, rank).unwrap();
                return (packed_merge & KEY_MASK, packed_merge, new_id);
            })
            .collect();

        let mphf = PtrHash::<PackedMerge, Linear>::new(
            keys.as_slice(),
            PtrHashParams {
                single_part: true,
                ..PtrHashParams::default_fast()
            },
        );

        let mut slots = vec![PackedMerge::new(0x00); len];
        let mut new_ids = vec![0u32; len];

        for index in 0..keys.len() {
            let slot = mphf.index_single_part(&keys[index]);
            slots[slot] = values[index];
            new_ids[slot] = ids[index];
        }

        Self {
            mphf,
            new_ids: new_ids.into_boxed_slice(),
            slots: slots.into_boxed_slice(),
        }
    }
}

impl MergeLookup for MergeStore {
    #[inline]
    fn get(&self, pair: &(u32, u32)) -> Option<(u32, u32)> {
        if self.slots.is_empty() {
            return None;
        }

        let key = make_packed_merge(pair.0, pair.1, 0x00).unwrap();
        let slot = self.mphf.index_single_part(&key);
        let entry = self.slots[slot];

        if entry & KEY_MASK != key {
            // PtrHash returns a valid slot for any u64
            // out-of-vocab pair lands on some other pair's slot:
            // Checking the lowest 40 bits of the slot signals the hit or miss
            return None;
        }
        Some(((entry >> 40) as u32, self.new_ids[slot]))
    }
}
