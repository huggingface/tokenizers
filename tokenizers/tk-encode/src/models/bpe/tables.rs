use ahash::RandomState;
use ahash::{AHashMap, HashMap, HashSet};
use itertools::Itertools;
use ptr_hash::{FastPtrHash, PtrHashParams, hash::NoHash};
use std::{cmp, fmt};

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
              // the flag allows us to store mrl and mrr!
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
        let n_slots = mphf.max_index();
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
    unmap: Box<[u32]>,      // unmap[internal_id] -> external_id
    pair_table: MphfMap,    // MPHF! because memory efficiency + bitwise makes check not costly
    top_merges: Box<[u64]>, // top 512 by 512 merges
    fold: Box<[u32]>,       // Which alphabet chars/bytes fold and can be merged directly
}

// byte level needs to unmap from non printable to the actual byte

impl BpeTables {
    pub(crate) fn build(vocab: AHashMap<String, u32>, merges: MergeMap) -> Self {
        // 1. We build the internal id map. This sorts the merges by their ranks so frequent pairs
        //    get a smaller rank.
        // used to build fold
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

        // BUILD internal map
        let mut top_merges = vec![u64::MAX; 512 * 512];
        let mut internal_id_map =
            vec![u32::MAX; *vocab.values().max().unwrap_or(&0u32) as usize + 1];
        let mut unmap = vec![u32::MAX; base + merges.len()];
        unmap[0..base].copy_from_slice(&alphabet);
        unmap[0..base]
            .iter()
            .enumerate()
            .for_each(|(a, b)| internal_id_map[*b as usize] = a as u32);
        let mut values = Vec::new();
        let mut keys = Vec::new();
        for ((a, b), (rank, external)) in merges.iter() {
            // the first spots are for the alphabet
            let internal = base as u32 + rank;
            unmap[internal as usize] = *external;
            internal_id_map[*external as usize] = internal;
            // a and b must already be in the map as they are part of the alphabet, or rank<
            let ia = internal_id_map[*a as usize];
            let ib = internal_id_map[*b as usize];
            let value = (*rank as u64) << 32 | internal as u64;
            // if a and b < 512 -> Dense grid
            if (ia | ib) < 512 {
                top_merges[(ia << 9 | ib) as usize] = value;
            } else {
                keys.push((ia, ib));
                values.push(value);
            }
        }
        let internal_id_map = internal_id_map.into_boxed_slice();
        let unmap = unmap.into_boxed_slice();
        let top_merges = top_merges.into_boxed_slice();
        let pair_table = MphfMap::build(keys, values);

        let cp_to_internal_id = build_conversion_table(vocab, merges, &internal_id_map, &unmap);
        let fold = cp_to_internal_id.into_boxed_slice();
        // Now let's build the MPHF for the merge pair table. The key is already a u64.
        // Slot is key as u64,
        // TODO: we need to add a log here on number of folder tokens, unique product merges, etc.
        Self {
            unmap,
            pair_table,
            top_merges,
            fold,
        }
    }
}

fn bytes_to_unicode() -> [char; 256] {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32)
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut cs: Vec<u32> = bs.clone();
    let mut n = 0;
    for b in 0u32..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut table = [' '; 256];
    for (b, c) in bs.iter().zip(cs.iter()) {
        table[*b as usize] = char::from_u32(*c).unwrap();
    }
    table
}

fn build_conversion_table(
    vocab: AHashMap<String, u32>,
    merges: AHashMap<(u32, u32), (u32, u32)>,
    internal_id_map: &Box<[u32]>,
    unmap: &Box<[u32]>,
) -> Vec<u32> {
    let mut merge_rank_left = vec![u32::MAX; merges.len()];
    let mut merge_rank_right = vec![u32::MAX; merges.len()];
    // We are building mrl and mrr which for a byte will tell us the minimum
    // rank of merge that involves it on the right or on the left. This allows us to check
    // for a char: b0,b1,b2 if its safe to fold. It is if:
    // - rank(b0, b1) <= rank(b0, *)
    // - rank(b1, b2) <= rank(b0, b1)
    // - rank(b2, *)  >= rank((b0,b1), b2)
    // - rank((b0,b1), b2) <= rank(b2, *)
    for (pair, key) in merges.iter() {
        merge_rank_right[pair.0 as usize] = cmp::min(merge_rank_left[pair.0 as usize], key.0);
        merge_rank_left[pair.1 as usize] = cmp::min(merge_rank_right[pair.0 as usize], key.0);
    }
    let mut non_bmp: AHashMap<char, u32> = AHashMap::new();
    let mut cp_to_internal_id = vec![u32::MAX; 65536];
    // BUILD the codepoint to internal id. This table will also account for byte level that are
    // safe to fold. b0|b1|b2 are safe to fold if rank(b0,b1) < rank(b0, *) & < rank(*, b1)
    // We cover the basic multilingual plan here, so any input codepoint that is <u32::MAX;
    // The rest of them have to be converted directly.
    let mut inv_table = Vec::new();
    let b2u = bytes_to_unicode();
    for b in 0..255 {
        inv_table[b2u[b] as usize] = b;
    }
    for (mut s, cp) in vocab {
        // 1. Filter string that are valid codepoints:
        // string tokens can be bytes as str in which case the count will be wrong.
        // thus we make sure to map them to the actual byte, and re-convert to utf8.
        let bytes = s
            .chars()
            .map(|ch| inv_table[ch as usize] as u8)
            .collect::<Vec<u8>>();
        // TODO: i don't even need the checks

        if let Ok(s) = str::from_utf8(&bytes) {
            if s.chars().count() > 1 {
                continue;
            }
        } else {
            continue;
        };
        // for each, we need to write at the codepoint the internal id.
        // we also have to check if its foldable.
        // fold will be indexed by cp (non utf8)
        // to set the value to internal token means we can safely convert 1,2 or 3 bytes to
        // internal id. This is true iff the bytes that compose is
        // here the codepoint could be multibyte and collapse to a single token. We do
        // pre-emptive merge instead of ByteLevel, iff (r < mrr[left_edge] && r < mrl[right_edge])r
        let mut buff = [u8::MAX; 4];
        let mut running_ids: Vec<u32> = bytes
            .iter()
            .map(|&byte| internal_id_map[usize::from(byte)])
            .collect();
        let mut safe = true;
        let mut foldable = false;
        loop {
            // are the bytes mergeable?
            let ib0 = running_ids[0];
            let ib1 = running_ids[1];
            let ibl = running_ids[running_ids.len() - 1];

            // merges does not use the internal rank but the external
            if let Some((r, id)) = merges.get(&(unmap[ib0 as usize], unmap[ib1 as usize])) {
                // if this fails, its unsafe to merge
                if merge_rank_right[ib0 as usize] >= *r && merge_rank_left[ibl as usize] >= *r {
                    running_ids[1] = internal_id_map[*id as usize];
                    running_ids = running_ids[1..].to_vec();
                } else {
                    safe = false;
                    break;
                }
            } else {
                break;
            }
            // is the rank merge the smallest?
            // is length of buff 1?
            if running_ids.len() == 1 {
                foldable = true;
                break;
            }
        }
        if safe {
            cp_to_internal_id[s.chars().next().unwrap() as usize] = internal_id_map[cp as usize];
        } else {
            // can't fold, we just convert to internal id for each byte
            for (i, b) in s.bytes().enumerate() {
                if cp_to_internal_id[usize::from(b)] == u32::MAX {
                    cp_to_internal_id[usize::from(b)] = running_ids[i];
                }
            }
        }
        log!(
            log::Level::Info,
            "Computed {:} foldable and {:} safe foldable bytes to chars",
            foldable,
            safe
        );
        // if *ch as u32 > 0xFFFF {
        //     non_bmp.insert(*ch, internal_id_map[*ch as usize]);
        // }
    }
    cp_to_internal_id
}

#[cfg(test)]
mod test {
    use ahash::AHashMap;

    use crate::models::bpe::{
        MergeMap,
        tables::{BpeTables, MphfMap, build_conversion_table},
    };
    #[test]
    pub fn test_build_conversion_table() {
        // we are gonna simulate byte-level merges
        let vocab = AHashMap::from_iter(vec![
            ("a".to_string(), 0),
            ("b".to_string(), 1),
            ("c".to_string(), 2),
            ("ab".to_string(), 3),
            ("aba".to_string(), 4),
            ("ba".to_string(), 5),
        ]);
        let mut merges = MergeMap::new();
        // keys are rank id, new id
        merges.insert((0, 1), (1, 3)); // a , b -> ab
        merges.insert((3, 0), (4, 4)); // ab, a -> aba
        merges.insert((1, 0), (3, 5)); // b , a -> ba  with rank(ab)  < rank(ba)
        merges.insert((3, 2), (2, 4)); // ab, c -> abc with rank(abc) < rank(aba)

        let out = build_conversion_table(
            vocab,
            merges,
            // we don't need complicated mapping so this one is just ordered
            &vec![0, 1, 2, 3, 4, 5].into_boxed_slice(),
            &vec![0, 1, 2, 3, 4, 5].into_boxed_slice(),
        );

        // test that 'aba' is not merged because 'abc' would have priority
        // but we want aba to be folded. But CP needs to be a codepoint to a 2-byte char
        assert_eq!(out['a' as usize], 0);
    }

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
        let tables = BpeTables::build(vocab, merges);
        // there are only 4 elements because ab and aba are part of the vocab
        // so the alphabet is a,b and the ranks are ab and aba
        assert_eq!(tables.pair_table.get(0u64 << 32 | 1u64) & 0xFFFF, 2u64);
    }
}
