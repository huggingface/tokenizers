use ahash::RandomState;
use ahash::{AHashMap, HashMap, HashSet};
use itertools::Itertools;
use ptr_hash::{FastPtrHash, PtrHashParams, hash::NoHash};
use std::fmt;

type Mphf = FastPtrHash<NoHash, u64>;

use crate::models::bpe::MergeMap;

// We built tables at load time based on the vocab and merges.
// There are 5 different tables:
// - Internal IDS: stores the byte levels and characters in their vocab order, and then we store
// the merges in their rank orders. This allows us to build the other tables at a lower cost, and
// converting back is almost free. This allows us to no longer carry rank and ID at the same time,
// and just look at ranks.
// - Pair table: for each merge pair (u64 packed key) we store key << 18 | new_id . The key is
// stored to check. This is a custom implementation of AHashmap to have a single load.
// - Grid: [u32; 1024, 1024] this is a dense merge for internal ids < 1024. Since we sort internal
// ids, this is the most used grid and only works because we sort the internal ids based on merge rank.
// - Participation bitmaps: 2 bools, true if  participates, one map for left, one for right. This
// allows to skip fast folded/chars that never actually participate in merges. This is used before
// checking the PairTable.
// - fold [u32; 65536]: this tables goes from codepoint to internal id directly. It is only
// adressable by the lvl1 codepoints, so basically characters / bytes.
//
// With this we implement the Lookup functions wich redirects based on the id comparisons.
//
//

// PairTable slot
#[derive(Clone)]
#[repr(C, align(16))]
struct Slot {
    key: u64, // holds (a << 32, b)
    val: u64, // holds rank as u64 << 32, flags << 30, id there is 2^30 possible ids, 1B is enough
}
// Fixed seeds so a given vocab always hashes identically (the hasher is also stored on the struct,
// so build and query are guaranteed consistent regardless).
const SEEDS: [u64; 4] = [
    0x243F_6A88_85A3_08D3,
    0x1319_8A2E_0370_7344,
    0xA409_3822_299F_31D0,
    0x082E_FA98_EC4E_6C89,
];

struct MphfMap {
    mphf: Mphf,
    hasher: RandomState,
    entries: Box<[Slot]>,
    /// `id_to_slot[token_id] -> entry_idx` -> index into entries as the entries are not really sorted.
    id_to_slot: Box<[u32]>,
    /// Number of real tokens. Cached at build so `len()` is O(1): `entries` is sized to the
    /// MPHF's non-minimal slot range (with phantom padding slots), so its length is not the
    /// token count.
    n: usize,
}

impl MphfMap {
    pub fn build(keys: Vec<(u32, u32)>, values: Vec<u64>) -> Self {
        let n = keys.len();

        let hasher = RandomState::with_seeds(SEEDS[0], SEEDS[1], SEEDS[2], SEEDS[3]);

        // 1. Pre-hash token bytes -> u64 keys using near perfect hash func
        let keys: Vec<u64> = keys
            .iter()
            .map(|(a, b)| hasher.hash_one((a << 32 | b) as u64))
            .collect();

        // 2. A perfect hash needs distinct keys. Collisions are astronomically unlikely
        //    (~n^2/2^65); if one ever fires, switch the key type to u128. The byte check below makes
        //    a collision a correct miss at query time, but it would drop a token at build, so guard.
        // TODO: check for collisions.

        // 3. Build the (non-minimal) `FastPtrHash` via `PtrHashParams::default_fast()`; query with `.index()`.
        let params = PtrHashParams::default_fast();
        let mphf = Mphf::new(&keys, params);

        // FastPtrHash is non-minimal: `index()` may return a slot up to `max_index()` (>= n),
        // so `entries` must be sized to cover the whole slot range. Slots never written by the
        // build loop stay as the default `Entry { len: 0, .. }` (phantom/padding slots), which
        // enumeration/count paths filter out via `len > 0`.
        let n_slots = mphf.max_index();

        // 4. Place each token at its MPHF slot; build the slab and the id->slot reverse table.
        let mut entries = vec![
            Slot {
                key: 0u64,
                val: 0u64
            };
            n_slots
        ];
        let total_slots = *(keys.iter().max().unwrap_or(&0u64)) as usize + 1;
        let mut id_to_slot = vec![u32::MAX; total_slots];
        for id in &keys {
            let slot = mphf.index(&hasher.hash_one(id));
            let val = values[*id as usize];
            entries[slot] = Slot {
                key: *id,
                val: 0u64,
            };
            id_to_slot[*id as usize] = slot as u32;
        }

        Self {
            mphf,
            hasher,
            entries: entries.into_boxed_slice(),
            id_to_slot: id_to_slot.into_boxed_slice(),
            n,
        }
    }
    #[inline]
    // from the key pair, returns the rank, the flags and the new id.
    pub fn get(self, key: u64) -> Option<u64> {
        let slot = self.mphf.index(&key);
        let e = &self.entries[slot];
        if e.key == key {
            return Some(e.val);
        } else {
            return None;
        }
    }
}
pub(crate) struct BpeTables {
    internal_id_map: Box<[u32]>, // internal_id_map[external_id] -> internal_id
    unmap: Box<[u32]>,           // unmap[internal_id] -> external_id
    pair_table: Box<[Slot]>,     // MPHF! because memory efficiency + bitwise makes check not costly
    top_merges: Box<[u64]>,      // top 512 by 512 merges
    fold: Box<[u32]>,            // Which alphabet chars/bytes fold and can be merged directly
}

impl BpeTables {
    pub(crate) fn build(vocab: AHashMap<String, u32>, merges: MergeMap) -> Self {
        // 1. We build the internal id map. This sorts the merges by their ranks so frequent pairs
        //    get a smaller rank.
        let vocab_r = AHashMap::from_iter(vocab.iter().map(|(a, b)| (b, a)));
        let mut pair_table = Box::new([]);
        let mut top_merges = Box::new([]);
        // used to build fold
        let mut merge_rank_left = Box::new(vec![0u32; merges.len()]);
        let mut merge_rank_right = Box::new(vec![0u32; merges.len()]);
        let mut fold = Box::new([]);

        let rev_merge = merges
            .iter()
            .map(|(_, (_, id))| *id)
            .collect::<HashSet<u32>>();

        let mut alphabet: Vec<u32> = vocab
            .values()
            .copied()
            .filter(|id| !rev_merge.contains(id))
            .collect();
        alphabet.sort_unstable();
        let base: usize = alphabet.len();

        let mut internal_id_map = vec![u32::MAX; *vocab.values().max().unwrap_or(&0u32) as usize];
        let mut unmap = vec![u32::MAX; base + merges.len()];
        unmap[0..base].copy_from_slice(&alphabet);
        unmap[0..base]
            .iter()
            .enumerate()
            .for_each(|(a, b)| internal_id_map[*b as usize] = a as u32);
        for (_, (rank, external)) in merges.iter() {
            // the first spots are for the alphabet
            let internal = base as u32 + rank;
            unmap[internal as usize] = *external;
            internal_id_map[*external as usize] = internal;
        }
        let internal_id_map = internal_id_map.into_boxed_slice();
        let unmap = unmap.into_boxed_slice();

        // Now let's build the MPHF for the merge pair table. The key is already a u64.
        // Slot is key as u64,
        // TODO: we need to add a log here on number of folder tokens, unique product merges, etc.
        Self {
            internal_id_map,
            unmap,
            pair_table,
            top_merges,
            fold,
        }
    }
}

#[cfg(test)]
mod test {
    use ahash::AHashMap;

    use crate::models::bpe::{MergeMap, tables::BpeTables};

    #[test]
    pub fn test_build() {
        let vocab = AHashMap::from_iter(vec![
            ("a".to_string(), 1),
            ("b".to_string(), 2),
            ("ab".to_string(), 5),
            ("ba".to_string(), 4),
            ("aab".to_string(), 3),
        ]);
        let mut merges = MergeMap::new();
        merges.insert((1, 2), (3, 1));
        merges.insert((1, 5), (4, 1));
        merges.insert((1, 3), (5, 1));
        println!("merges: {:?}", merges);
        let tables = BpeTables::build(vocab, merges);
        assert_eq!(tables.internal_id_map.to_vec(), vec![0, 1, 3, 4, 5]);
    }
}
