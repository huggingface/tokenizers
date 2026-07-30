use ahash::RandomState;
use ahash::{AHashMap, HashSet};
use ptr_hash::{FastPtrHash, PtrHashParams, hash::NoHash};
use std::cmp;

type Mphf = FastPtrHash<NoHash, u64>;

use crate::models::bpe::MergeMap;
use crate::models::bpe::bytelevel_folding::{ByteLevelFold, Fold};

/// Pair-table value layout: `rank[63:32] | internal_id[31:0]`, sentinel `u64::MAX`. Rank is
/// shifted to the high half so `val < min_val` is a rank comparison without having to do any
/// shifting.

// We built tables at load time based on the vocab and merges.
// There are 5 different tables:
// - Internal IDS: stores the byte levels and characters in their vocab order, and then we store
// the merges in their rank orders. This allows us to build the other tables at a lower cost, and
// converting back is almost free. This allows us to no longer carry rank and ID at the same time,
// and just look at ranks. It also means more frequent merges can live in a L1 cache.
// - Pair table: for each merge pair (u64 packed key) we build a custom hash, close adressing
// for memory efficiency. The key is stored in the value to check.
// - Grid: [u32; 512*512] this is a dense merge for internal ids < 512. Since we sort rank ids, it
// holds the most frequent merges.
// - fold [u32; 65536]: this tables goes from codepoint (char) to internal id directly. It is the
// trickiest to build, especially for byte level tokenizer. We directly map 2-3 byte chars
// to the merged token if we can prove that BPE would construct it. We leverage boundaries (start
// bytes after end byte).
// - non_bmp: this holds a mapping from char to the index in the vocab when we can't fold. Hashing
// is slower and less efficient, but bmp are rare.

// PairTable slot
#[derive(Clone)]
#[repr(C, align(16))]
struct Slot {
    key: u64, // holds (a << 32, b)
    val: u64, // holds rank as u64 << 32, flags << 30, id there is 2^30 possible ids, 1B is enough
              // rank sits high so `val < min_val` is a rank comparison. mrl/mrr are NOT stored
              // here: they are build-time only, consumed by the fold guard.
}
// Fixed seeds so a given vocab always hashes identically (the hasher is also stored on the struct,
// so build and query are guaranteed consistent regardless).
const SEEDS: [u64; 4] = [
    0x243F_6A88_85A3_08D3,
    0x1319_8A2E_0370_7344,
    0xA409_3822_299F_31D0,
    0x082E_FA98_EC4E_6C89,
];

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
                val: val,
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
    #[inline]
    // from the key pair, returns the rank, the flags and the new id.
    pub fn get(&self, key: u64) -> u64 {
        let slot = self.mphf.index(&self.hasher.hash_one(key));
        let e = &self.entries[slot];
        if e.key == key {
            return e.val;
        } else {
            return u64::MAX;
        }
    }
}
pub(crate) struct BpeTables {
    pub unmap: Box<[u32]>,            // unmap[internal_id] -> external_id
    pub pair_table: MphfMap, // MPHF! because memory efficiency + bitwise makes check not costly
    pub top_merges: Box<[u64]>, // top 512 by 512 merges, same packed value as the pair table
    pub fold: Box<[u32]>,    // codepoint in vocab to internal id
    pub non_bmp: AHashMap<char, u32>, // same as `fold`, for the codepoints past 0xFFFF (emoji, CJK ext)
}

impl BpeTables {
    pub(crate) fn build(vocab: AHashMap<String, u32>, merges: MergeMap, byte_level: bool) -> Self {
        // 1. We build the internal id map. This sorts the merges by their ranks so frequent pairs
        //    get a smaller rank.
        let rev_merge = merges
            .iter()
            .map(|(_, (_, id))| *id)
            .collect::<HashSet<u32>>();

        // vocab tokens that are not obtained by any merge
        let mut alphabet: Vec<u32> = vocab
            .values()
            .copied()
            .filter(|id| !rev_merge.contains(id))
            .collect();
        alphabet.sort_unstable();
        let base: usize = alphabet.len();

        // Products (unique merges result obtainable from potentially many pairs) get one internal id for the LOWEST rank.
        // llama-3: 280_147 merges -> 127_744 distinct products). The internal ID only account for
        // them, not the duplicates. We compute the lowest rank of the different merge that give
        // the same product.
        let mut lowest_rank: AHashMap<u32, u32> = AHashMap::new();
        for (_, (rank, merge_id)) in merges.iter() {
            let slot = lowest_rank.entry(*merge_id).or_insert(*rank);
            *slot = cmp::min(*slot, *rank);
        }
        let mut products: Vec<(u32, u32)> = lowest_rank.iter().map(|(p, r)| (*r, *p)).collect();
        products.sort_unstable();

        // this one is destroyed afterwards, does not matter if its big.
        let mut internal_id_map =
            vec![u32::MAX; *vocab.values().max().unwrap_or(&0u32) as usize + 1];
        let mut unmap = vec![u32::MAX; base + products.len()];
        // fill the first 0->base with the alphabet sorted by rank.
        unmap[0..base].copy_from_slice(&alphabet);
        for (internal, external) in alphabet.iter().enumerate() {
            internal_id_map[*external as usize] = internal as u32;
        }
        // now fill the rest of the tables with products sorted by rank.
        for (pos, (_, product)) in products.iter().enumerate() {
            let internal = (base + pos) as u32;
            unmap[internal as usize] = *product;
            internal_id_map[*product as usize] = internal;
        }
        let (cp_to_internal_id, non_bmp) =
            build_conversion_table(&vocab, &merges, &internal_id_map, &unmap, byte_level);
        let fold = cp_to_internal_id.into_boxed_slice();

        let mut top_merges = vec![u64::MAX; 512 * 512];
        let mut values = Vec::new();
        let mut keys = Vec::new();
        let mut dropped = 0usize;
        for ((a, b), (rank, product)) in merges.iter() {
            let ia = internal_id_map
                .get(*a as usize)
                .copied()
                .unwrap_or(u32::MAX);
            let ib = internal_id_map
                .get(*b as usize)
                .copied()
                .unwrap_or(u32::MAX);
            if ia == u32::MAX || ib == u32::MAX {
                dropped += 1; // merge over a token that is not in the vocab: malformed file
                continue;
            }
            let internal = internal_id_map[*product as usize] as u64;
            let value = (*rank as u64) << 32 | internal;
            // if a and b < 512 -> Dense grid
            if (ia | ib) < 512 {
                top_merges[(ia << 9 | ib) as usize] = value;
            } else {
                keys.push((ia, ib));
                values.push(value);
            }
        }
        let unmap = unmap.into_boxed_slice();
        let top_merges = top_merges.into_boxed_slice();
        let pair_table = MphfMap::build(keys, values);
        info!(
            "bpe tables: {base} alphabet + {} products (unique merges), {} merge in the dense grid, {dropped} merges dropped",
            products.len(),
            512 * 512 - top_merges.iter().filter(|c| **c == u64::MAX).count()
        );
        Self {
            unmap,
            pair_table,
            top_merges,
            fold,
            non_bmp,
        }
    }
    pub fn get_value(&self, a: &u32, b: &u32) -> u64 {
        if (a | b) < 512 {
            return self.top_merges[(a << 9 | b) as usize];
        } else {
            return self.pair_table.get(((*a as u64) << 32) | *b as u64);
        }
    }
}

/// We build the codepoint character to internal id table.
fn build_conversion_table(
    vocab: &AHashMap<String, u32>,
    merges: &MergeMap,
    internal_id_map: &[u32],
    unmap: &[u32],
    byte_level: bool,
) -> (Vec<u32>, AHashMap<char, u32>) {
    // We don't create a hashmap for everything for memory efficiency.
    fn place(bmp: &mut [u32], non_bmp: &mut AHashMap<char, u32>, ch: char, id: u32) {
        if (ch as u32) < 0x10000 {
            bmp[ch as usize] = id;
        } else {
            non_bmp.insert(ch, id);
        }
    }

    let mut cp_to_internal_id = vec![u32::MAX; 65536];
    let mut non_bmp: AHashMap<char, u32> = AHashMap::new();
    let (mut folded, mut unsafe_chars) = (0usize, 0usize);
    if byte_level {
        // A character reaches the merge loop as bytes, so folding it means proving the
        // merges are  predetermined. See `bytelevel_folding`.
        let folder = ByteLevelFold::new(vocab, merges, internal_id_map, unmap);
        for (s, external) in vocab.iter() {
            match folder.fold(s, *external) {
                Fold::Folds(ch, id) => {
                    place(&mut cp_to_internal_id, &mut non_bmp, ch, id);
                    folded += 1;
                }
                Fold::Unsafe => unsafe_chars += 1,
                Fold::Skip => {}
            }
        }
    } else {
        // simple case, we just write the vocab tokens to a dense table instead of a HashMap.
        for (s, external) in vocab.iter() {
            let mut it = s.chars();
            if let (Some(ch), None) = (it.next(), it.next()) {
                let id = internal_id_map
                    .get(*external as usize)
                    .copied()
                    .unwrap_or(u32::MAX);
                place(&mut cp_to_internal_id, &mut non_bmp, ch, id);
                folded += 1;
            }
        }
    }
    info!("fold table: {folded} characters fold, {unsafe_chars} formable but boundary-unsafe");
    (cp_to_internal_id, non_bmp)
}

#[cfg(test)]
mod test {
    use ahash::AHashMap;

    use crate::models::bpe::{
        MergeMap,
        tables::{BpeTables, MphfMap},
    };
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
        let value = 1u64 << 32 | 5 as u64;
        assert_eq!(pair_table.get(1u64 << 32 | 2u64), value);
    }

    #[test]
    pub fn test_build() {
        let vocab = AHashMap::from_iter(vec![
            ("a".to_string(), 0),
            ("b".to_string(), 1),
            ("ab".to_string(), 2),
            ("aba".to_string(), 3),
        ]);
        let mut merges = MergeMap::new();
        merges.insert((0, 1), (0, 2));
        merges.insert((3, 0), (1, 3));
        let tables = BpeTables::build(vocab, merges, true);
        // there are only 4 elements because ab and aba are part of the vocab
        // so the alphabet is a,b and the ranks are ab and aba.
        // Both operands are < 512, so the merge lives in the dense grid, not the MPHF.
        // grid and pair table share the value layout, so both halves have to be right
        assert_eq!(tables.top_merges[1], 2u64); // (a, b) -> ab: rank 0, internal 2
        assert_eq!(tables.top_merges[3 << 9], 1u64 << 32 | 3); // (aba, a) -> aba: rank 1, internal 3
        assert_eq!(tables.top_merges[2], u64::MAX); // (a, c) is not a merge
        assert_eq!(tables.pair_table.get(1u64), u64::MAX); // and nowhere else
        assert_eq!(&*tables.unmap, &[0, 1, 2, 3]);
    }
}
