use ahash::{AHashMap, HashMap, HashSet};
use itertools::Itertools;

use crate::models::bpe::MergeMap;

// We built tables at load time based on the vocab and merges.
// There are 5 different tables:
// - Internal IDS: stores the byte levels and characters in their vocab order, and then we store
// the merges in their rank orders. This allows us to the other tables at a lower cost, and
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
pub(crate) struct BpeTables {
    internal_id_map: Box<[u32]>,
    unmap: Box<[u32]>,
    pair_table: Box<[u64]>,
    top_merges: Box<[u32]>,
    fold: Box<[u32]>,
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

        let mut internal_id_map = vec![0u32; base + merges.len()];
        let mut unmap = vec![0u32; base + merges.len()];
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
    }
}
