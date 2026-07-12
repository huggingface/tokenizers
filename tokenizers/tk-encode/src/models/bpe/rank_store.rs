//! MPHF-backed pair → (rank, merged_id) store for the BPE merge hot path.
//!
//! Replaces the `AHashMap<(u32,u32),(u32,u32)>` with a `ptr_hash` minimal
//! perfect hash over the packed `(left<<32)|right` pair keys. A lookup is:
//! MPHF index (pure compute, its tables stay in cache) → **one** 16-byte,
//! 16-aligned `Entry` read (a single cache line) → verify the key (rejects
//! out-of-vocab pairs, whose MPHF slot is arbitrary). Half the memory of the
//! HashMap (~16 B/entry: llama-3's ~280k merges ≈ 4.5 MB vs ~8.75 MB) and the
//! merge becomes compute-bound rather than DRAM-bound.

use super::model::MergeMap;
use ptr_hash::bucket_fn::Linear;
use ptr_hash::{PtrHash, PtrHashParams};

type Mphf = PtrHash<u64, Linear>;

#[inline(always)]
fn pack(a: u32, b: u32) -> u64 {
    ((a as u64) << 32) | b as u64
}

/// splitmix64 finalizer: a *bijection* u64→u64. Spreads the structured packed
/// key (high 32 = left id, low 32 = right id) into a well-distributed hash for
/// the MPHF, with zero collisions by construction (distinct pairs → distinct
/// packs → distinct mixes), so no dedup/collision handling is needed.
#[inline(always)]
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

#[repr(align(16))]
#[derive(Clone, Copy)]
struct Entry {
    key: u64,
    rank: u32,
    merged: u32,
}

pub(crate) struct RankStore {
    mphf: Mphf,
    entries: Box<[Entry]>,
}

impl RankStore {
    pub(crate) fn build(merges: &MergeMap) -> Self {
        let mut params = PtrHashParams::default_fast();
        params.single_part = true;
        if merges.is_empty() {
            let empty: [u64; 0] = [];
            return RankStore {
                mphf: Mphf::new(&empty, params),
                entries: Box::new([]),
            };
        }
        // Collect once so build order is stable; each merged id is unique so
        // every slot is written exactly once (image gotcha #5).
        let pairs: Vec<((u32, u32), (u32, u32))> =
            merges.iter().map(|(&p, &v)| (p, v)).collect();
        let keys: Vec<u64> = pairs.iter().map(|&((a, b), _)| mix(pack(a, b))).collect();
        let mphf = Mphf::new(&keys, params);
        let n = pairs.len();
        let mut entries = vec![
            Entry { key: u64::MAX, rank: 0, merged: 0 };
            n
        ]
        .into_boxed_slice();
        for &((a, b), (rank, merged)) in &pairs {
            let k = pack(a, b);
            let slot = mphf.index_single_part(&mix(k));
            entries[slot] = Entry { key: k, rank, merged };
        }
        debug_assert!(
            entries.iter().all(|e| e.key != u64::MAX),
            "RankStore mis-sized: a slot was never written"
        );
        RankStore { mphf, entries }
    }

    /// One MPHF compute + one aligned `Entry` read + key verify. `None` for
    /// out-of-vocab pairs (the MPHF returns an arbitrary slot for those; the
    /// key check is what rejects them — not optional).
    #[inline]
    pub(crate) fn get(&self, a: u32, b: u32) -> Option<(u32, u32)> {
        if self.entries.is_empty() {
            return None;
        }
        let key = pack(a, b);
        let slot = self.mphf.index_single_part(&mix(key));
        let e = self.entries[slot];
        if e.key == key {
            Some((e.rank, e.merged))
        } else {
            None
        }
    }
}
