use ahash::RandomState;
use ahash::{AHashMap, HashSet};
use ptr_hash::{FastPtrHash, PtrHashParams, hash::NoHash};
use std::cmp;

type Mphf = FastPtrHash<NoHash, u64>;

use crate::models::bpe::MergeMap;
use crate::models::bpe::bytelevel_folding::{ByteLevelFold, Fold};

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
    unmap: Box<[u32]>,            // unmap[internal_id] -> external_id
    pair_table: MphfMap, // MPHF! because memory efficiency + bitwise makes check not costly
    top_merges: Box<[u64]>, // top 512 by 512 merges
    fold: Box<[u32]>,    // Which alphabet chars/bytes fold and can be merged directly
    non_bmp: AHashMap<char, u32>, // same as `fold`, for the codepoints past 0xFFFF (emoji, CJK ext)
}

// byte level needs to unmap from non printable to the actual byte

impl BpeTables {
    pub(crate) fn build(vocab: AHashMap<String, u32>, merges: MergeMap, byte_level: bool) -> Self {
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

        // BUILD internal map.
        // Products get one internal id each, canonicalised on their LOWEST rank. `base + rank`
        // only holds for strictly 1:1 vocabs; converted ones reuse a product across several
        // merges (llama-3: 280_147 merges -> 127_744 distinct products), so ranks are not a
        // dense id space and two ids for one token would break `unmap`.
        let mut lowest_rank: AHashMap<u32, u32> = AHashMap::new();
        for (_, (rank, product)) in merges.iter() {
            let slot = lowest_rank.entry(*product).or_insert(*rank);
            *slot = cmp::min(*slot, *rank);
        }
        let mut products: Vec<(u32, u32)> = lowest_rank.iter().map(|(p, r)| (*r, *p)).collect();
        products.sort_unstable();

        let mut internal_id_map =
            vec![u32::MAX; *vocab.values().max().unwrap_or(&0u32) as usize + 1];
        let mut unmap = vec![u32::MAX; base + products.len()];
        unmap[0..base].copy_from_slice(&alphabet);
        for (internal, external) in alphabet.iter().enumerate() {
            internal_id_map[*external as usize] = internal as u32;
        }
        for (pos, (_, product)) in products.iter().enumerate() {
            let internal = (base + pos) as u32;
            unmap[internal as usize] = *product;
            internal_id_map[*product as usize] = internal;
        }

        // Only now is every operand resolvable: `merges.iter()` is hash order, so a merge whose
        // operand is another merge's product would otherwise read u32::MAX and push a garbage key.
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
            let value = (*rank as u64) << 32 | internal_id_map[*product as usize] as u64;
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

        let (cp_to_internal_id, non_bmp) =
            build_conversion_table(vocab, merges, &internal_id_map, &unmap, byte_level);
        let fold = cp_to_internal_id.into_boxed_slice();
        info!(
            "bpe tables: {base} alphabet + {} products, {} in the dense grid, {dropped} merges dropped",
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
}

/// `byte_level` says which alphabet the vocab keys are written in, and the two readings are
/// incompatible: `"Ġ"` is byte 0x20 remapped when it is true and the character U+0120 when it is
/// false. Nothing in the vocab itself distinguishes them, so the caller has to say.
fn build_conversion_table(
    vocab: AHashMap<String, u32>,
    merges: MergeMap,
    internal_id_map: &[u32],
    unmap: &[u32],
    byte_level: bool,
) -> (Vec<u32>, AHashMap<char, u32>) {
    // an emoji is a single-char token there, but four remapped
    // chars under byte-level, so it can never be one token's worth of codepoint.
    fn place(bmp: &mut [u32], non_bmp: &mut AHashMap<char, u32>, ch: char, id: u32) {
        if (ch as u32) < 0x10000 {
            bmp[ch as usize] = id;
        } else {
            non_bmp.insert(ch, id);
        }
    }

    // BUILD the codepoint to internal id. Covers the BMP directly; past 0xFFFF a 4 MB table
    // is not worth it, so those go in a map.
    let mut cp_to_internal_id = vec![u32::MAX; 65536];
    let mut non_bmp: AHashMap<char, u32> = AHashMap::new();
    let (mut folded, mut unsafe_chars) = (0usize, 0usize);
    if byte_level {
        // A character reaches the merge loop as its bytes, so folding it means proving the
        // assembly is predetermined. See `bytelevel_folding`.
        let folder = ByteLevelFold::new(&vocab, &merges, internal_id_map, unmap);
        for (s, external) in vocab.iter() {
            match folder.fold(s, *external) {
                Fold::Folds(ch, id) => {
                    place(&mut cp_to_internal_id, &mut non_bmp, ch, id);
                    folded += 1;
                }
                Fold::Unsafe => unsafe_chars += 1,
                // No entry at all. The u32::MAX sentinel makes the encoder emit the character's
                // bytes and let the merge loop assemble them, which is always exact.
                Fold::Skip => {}
            }
        }
    } else {
        // Char mode: a single-character token IS an atom, exactly what reference BPE starts
        // from, so there is no byte assembly to replay and no edge for a neighbour to steal.
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

        let (out, _) = build_conversion_table(
            vocab,
            merges,
            // we don't need complicated mapping so this one is just ordered
            &vec![0, 1, 2, 3, 4, 5].into_boxed_slice(),
            &vec![0, 1, 2, 3, 4, 5].into_boxed_slice(),
            true,
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
        let tables = BpeTables::build(vocab, merges, true);
        // there are only 4 elements because ab and aba are part of the vocab
        // so the alphabet is a,b and the ranks are ab and aba.
        // Both operands are < 512, so the merge lives in the dense grid, not the MPHF.
        assert_eq!(tables.top_merges[1] & 0xFFFF_FFFF, 2u64); // (a, b) -> ab, internal 2
        assert_eq!(tables.top_merges[1] >> 32, 0u64); // at rank 0
        assert_eq!(tables.pair_table.get(1u64), u64::MAX); // and nowhere else
        assert_eq!(&*tables.unmap, &[0, 1, 2, 3]);
    }
}

#[cfg(test)]
mod real_vocab_test {
    use super::{BpeTables, MergeMap};
    use ahash::AHashMap;

    fn load(path: &str, vocab_key: &str) -> (AHashMap<String, u32>, MergeMap) {
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let root = if vocab_key.is_empty() {
            &json
        } else {
            &json[vocab_key]
        };
        let vocab: AHashMap<String, u32> = root["vocab"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.as_u64().unwrap() as u32))
            .collect();
        let merges: MergeMap = root["merges"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .filter_map(|(rank, x)| {
                let (a, b) = if let Some(s) = x.as_str() {
                    let (a, b) = s.split_once(' ').unwrap();
                    (a.to_string(), b.to_string())
                } else {
                    let arr = x.as_array().unwrap();
                    (
                        arr[0].as_str().unwrap().to_string(),
                        arr[1].as_str().unwrap().to_string(),
                    )
                };
                // The slim fixtures carry more merges than vocab, so a merge can name a token
                // that does not exist. Keep those out of `merges` here and the ones that only
                // lose an operand exercise the `dropped` path in `build`.
                let product = *vocab.get(&format!("{a}{b}"))?;
                Some((
                    (*vocab.get(&a)?, *vocab.get(&b).unwrap_or(&u32::MAX)),
                    (rank as u32, product),
                ))
            })
            .collect();
        (vocab, merges)
    }

    fn report(name: &str, path: &str, key: &str, byte_level: bool) {
        if !std::path::Path::new(path).exists() {
            println!("{name}: SKIPPED, {path} not present");
            return;
        }
        let (vocab, merges) = load(path, key);
        let n_vocab = vocab.len();
        let n_merges = merges.len();
        let products: std::collections::HashSet<u32> = merges.values().map(|(_, p)| *p).collect();
        let t = BpeTables::build(vocab, merges, byte_level);
        let folded = t.fold.iter().filter(|v| **v != u32::MAX).count();
        let ascii = t.fold[0..128].iter().filter(|v| **v != u32::MAX).count();
        println!(
            "{name}: vocab {n_vocab}, merges {n_merges} -> {} products, unmap {}, \
             fold {folded} ({ascii} ascii + {} multi-byte), non_bmp {}",
            products.len(),
            t.unmap.len(),
            folded - ascii,
            t.non_bmp.len(),
        );
        assert!(
            t.unmap.iter().all(|v| *v != u32::MAX),
            "{name}: unmap has holes"
        );
    }

    /// `byte_level = false` is the char-mode arm: vocab keys are raw text, so single-char
    /// tokens fold directly and codepoints past the BMP land in `non_bmp` (gemma: 2306).
    /// The two char-mode vocabs are not in-tree, so they skip when absent.
    #[test]
    fn real_vocabs() {
        let hub = format!(
            "{}/.cache/huggingface/hub",
            std::env::var("HOME").unwrap_or_default()
        );
        for (name, path, byte_level) in [
            ("gpt2", "../data/gpt2.json".to_string(), true),
            ("deepseek", "../data/deepseek-v4.json".to_string(), true),
            (
                "llama-3",
                "../data/llama-3-tokenizer.json".to_string(),
                true,
            ),
            ("glm-5.2", "../data/glm-5.2-slim.json".to_string(), true),
            ("gpt-oss", "../data/gpt-oss-slim.json".to_string(), true),
            (
                "llama-2",
                format!(
                    "{hub}/models--meta-llama--Llama-2-7b-hf/snapshots/\
                     01c7f73d771dfac7d292323805ebc428287df4f9/tokenizer.json"
                ),
                false,
            ),
            (
                "gemma-3",
                format!(
                    "{hub}/models--google--gemma-3-4b-it/snapshots/\
                     093f9f388b31de276ce2de164bdc2081324b9767/tokenizer.json"
                ),
                false,
            ),
        ] {
            report(name, &path, "model", byte_level);
        }
    }
}
