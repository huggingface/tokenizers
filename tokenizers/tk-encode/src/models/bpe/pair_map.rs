//! Merge lookup for pairs outside the dense grid: a perfect-hash map from a packed
//! `(left, right)` pair of internal ids to the pair's packed merge value. The `tables` module
//! documents the value layout.
use ahash::RandomState;
use ptr_hash::{FastPtrHash, PtrHashParams, hash::NoHash};
use std::cmp;

type Mphf = FastPtrHash<NoHash, u64>;

// Fixed seeds so a given vocab always hashes identically (the hasher is also stored on the struct,
// so build and query are guaranteed consistent regardless).
const SEEDS: [u64; 4] = [
    0x243F_6A88_85A3_08D3,
    0x1319_8A2E_0370_7344,
    0xA409_3822_299F_31D0,
    0x082E_FA98_EC4E_6C89,
];

#[derive(Clone)]
#[repr(C, align(16))]
struct Slot {
    key: u64, // holds (a << 32 | b), u64::MAX when the slot is empty
    val: u64, // the pair's packed merge value
}

pub struct MphfMap {
    mphf: Mphf,
    hasher: RandomState,
    entries: Box<[Slot]>,
}

impl MphfMap {
    pub fn build(keys: Vec<(u32, u32)>, values: Vec<u64>) -> Self {
        assert!(
            keys.len() == values.len(),
            "Keys and values must be of same lengths"
        );
        let hasher = RandomState::with_seeds(SEEDS[0], SEEDS[1], SEEDS[2], SEEDS[3]);

        // 1. Pre-hash token bytes -> u64 keys using near perfect hash func
        let h_keys: Vec<u64> = keys
            .iter()
            .map(|(a, b)| hasher.hash_one((*a as u64) << 32 | *b as u64))
            .collect();

        // 2. A perfect hash needs distinct keys. Collisions are astronomically unlikely
        //    (~n^2/2^65); if one ever fires, switch the key type to u128. The byte check below makes
        //    a collision a correct miss at query time, but it would drop a token at build, so guard.
        // TODO: check for collisions.

        // 3. Build the (non-minimal) `FastPtrHash` via `PtrHashParams::default_fast()`; query with `.index()`.
        let params = PtrHashParams::default_fast();
        let mphf = Mphf::new(&h_keys, params);
        // At least one slot: a small vocab can have every merge inside the dense grid, and an
        // empty slab would make `get` index out of bounds. u64::MAX is never a real key (that
        // needs both operands to be u32::MAX), so the lone slot always misses.
        let n_slots = cmp::max(mphf.max_index(), 1);
        // 4. Place each token at its MPHF slot; build the slab and the id->slot reverse table.
        let mut entries = vec![
            Slot {
                key: u64::MAX,
                val: u64::MAX
            };
            n_slots
        ];
        for (pos, (a, b)) in keys.iter().enumerate() {
            let hash = h_keys[pos];
            let slot = mphf.index(&hash);
            let val = values[pos];
            entries[slot] = Slot {
                key: (*a as u64) << 32 | *b as u64,
                val,
            };
        }

        let new = Self {
            mphf,
            hasher,
            entries: entries.into_boxed_slice(),
        };

        for (k, v) in keys.iter().zip(values) {
            // we check that we keys and values were properly sorted
            assert_eq!(
                new.get((k.0 as u64) << 32 | k.1 as u64),
                v,
                "The values stored for one of the keys is wrong. This probably means a wrong index in values"
            );
        }
        new
    }
    /// Every live entry, as `(packed key, packed value)`.
    ///
    /// Slot order, which is arbitrary -- a caller that wants the merge list back sorts by the rank
    /// in the value's high half. Enumerable at all because each slot stores its own key for the
    /// query-time check, so the perfect hash never has to be inverted.
    pub(super) fn iter(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        // `u64::MAX` needs both operands to be `u32::MAX`, so it is never a real key -- the same
        // fact `build` relies on to leave unused slots empty.
        self.entries
            .iter()
            .filter(|e| e.key != u64::MAX)
            .map(|e| (e.key, e.val))
    }

    #[inline]
    // from the key pair, returns the rank, the flags and the new id.
    pub fn get(&self, key: u64) -> u64 {
        let slot = self.mphf.index(&self.hasher.hash_one(key));
        let e = &self.entries[slot];
        if e.key == key { e.val } else { u64::MAX }
    }
}

#[cfg(test)]
mod test {
    use crate::models::bpe::{MergeMap, pair_map::MphfMap};

    #[test]
    pub fn test_mphf() {
        let mut merges = MergeMap::new();
        merges.insert((1, 2), (1, 5));
        merges.insert((1, 5), (4, 1));

        let (keys, values): (Vec<(u32, u32)>, Vec<u64>) = merges
            .iter()
            .map(|((a, b), (rank, id))| ((*a, *b), (*rank as u64) << 32 | (*id as u64)))
            .unzip();
        let pair_table = MphfMap::build(keys, values);
        let value = 1u64 << 32 | 5_u64;
        assert_eq!(pair_table.get(1u64 << 32 | 2u64), value);
    }
}
