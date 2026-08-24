//! The lookup tables the merge engines run on, built once at load time from the vocab and merges.
//!
//! # Internal ids
//!
//! The vocab's own ids ("external") are remapped to internal ids: first the alphabet (tokens that
//! no merge produces) in vocab order, then every merge product in rank order. Frequent merges get
//! small ids, so they land in the dense grid and stay hot in cache, and the engines can compare
//! ranks without carrying rank and id side by side. `unmap` takes an internal id back to the
//! external one; the model applies it once, when a word is done merging.
//!
//! # The packed merge value
//!
//! Every lookup answers the same question, does this pair merge and into what, with one `u64`:
//!
//! ```text
//! bit 63                              32   31     30   29                           0
//!    ┌──────────────────────────────────┬──────┬────┬──────────────────────────────┐
//!    │            rank : u32            │unused│SAFE│      product id : 30 bits    │
//!    │      (merge priority, 0 = best)  │      │    │   (internal id of the token  │
//!    │                                  │      │    │      this pair merges into)  │
//!    └──────────────────────────────────┴──────┴────┴──────────────────────────────┘
//! ```
//!
//! `u64::MAX` means the pair does not merge. Rank sits in the high half so comparing two whole
//! values is a rank comparison without any shifting. The `min_rank_left`/`min_rank_right` arrays
//! computed below are not part of the value: they are build-time only, consumed by the fold
//! guard (see `fold::byte_level`) and by the SAFE bit.
//!
//! # Where a pair is looked up
//!
//! [`BpeTables::get_value`] tries two places: the dense grid (`top_index`/`top_values`) when both
//! ids are < 512, which holds the most frequent merges thanks to the rank ordering, and the
//! perfect-hash [`MphfMap`] for every other pair. Text is converted to its first symbols before
//! any of this, through [`SparseFold`] and `byte_internal` (see the `convert` module).
use ahash::{AHashMap, HashSet};
use std::cmp;

use crate::models::bpe::fold::{self, SparseFold};
use crate::models::bpe::pair_map::MphfMap;
use crate::models::bpe::{At, MergeMap};

/// The rank half of a packed merge value, for reusing a rank as the high half of a queue key.
/// It keeps the rank alone: everything below bit 32 is dropped, flags and product id together, and
/// an unmergeable pair (`u64::MAX`) still masks to a rank of `u32::MAX`, the worst possible.
pub(super) const RANK_MASK: u64 = 0xFFFF_FFFF_0000_0000;

/// The product-id half, which is the low 30 bits: bits 30 and 31 are the flag field, so every read
/// of a product id masks rather than truncating to `u32`. 2^30 ids is ~1.07 B, far past any vocab.
pub(super) const ID_MASK: u64 = (1 << 30) - 1;

/// Bit 30: batching every occurrence of this pair in one multipass sweep is exact. It is only so
/// when the product cannot reach a merge cheaper than the one being applied,
/// `rank < min(min_rank_left[product], min_rank_right[product])` -- otherwise that cheaper merge is
/// due before the pair's remaining occurrences, and the sweep has to stop at the first one.
/// gpt2 and deepseek have no unsafe merges at all; llama-2 and llama-3 have ~22%.
pub(super) const SAFE_MASK: u64 = 1 << 30;

pub(crate) struct BpeTables {
    pub unmap: Box<[u32]>,   // unmap[internal_id] -> external_id
    pub pair_table: MphfMap, // MPHF! because memory efficiency + bitwise makes check not costly
    /// The 512x512 grid of hottest pairs, kept directly indexed so a lookup is one load, but with
    /// a u16 index per cell instead of the value inline: only 3.5-5.7% of cells hold a merge, so
    /// 2 MiB of u64s becomes 512 KB of indices plus 8 B per live entry. A miss is still one load, a
    /// hit is two.
    pub top_index: Box<[u16]>,
    pub top_values: Box<[u64]>,
    pub fold: SparseFold, // codepoint in vocab to internal id, sparse: see SparseFold
    pub byte_internal: [u32; 256], // byte -> internal id, for characters that do not fold
    /// False when every merge is safe, which lets multipass skip the per-pass SAFE test entirely.
    pub any_unsafe: bool,
}

impl BpeTables {
    /// Returns the tables plus the dense `external id -> internal id` map built along the way.
    /// Callers that do not need the map just drop it; it is ~4 bytes per vocab entry.
    pub(crate) fn build(
        vocab: AHashMap<String, u32>,
        merges: MergeMap,
        byte_level: bool,
    ) -> (Self, Vec<u32>) {
        // 1. We build the internal id map. This sorts the merges by their ranks so frequent pairs
        //    get a smaller rank.
        let rev_merge = merges
            .iter()
            .map(|(_, (_, id))| *id)
            .collect::<HashSet<u32>>();

        // vocab tokens that are not obtained by any merge. Held as u64 rather than u32 purely so
        // that this sort is the same `[u64]` instantiation as every other load-time sort: each
        // distinct key type costs its own ~2 KB copy of driftsort/ipnsort in the binary.
        let mut alphabet: Vec<u64> = vocab
            .values()
            .copied()
            .filter(|id| !rev_merge.contains(id))
            .map(|id| id as u64)
            .collect();
        alphabet.sort_unstable();
        let base: usize = alphabet.len();

        // Products (unique merges result obtainable from potentially many pairs) get one internal id for the LOWEST rank.
        // llama-3: 280_147 merges -> 127_744 distinct products). The internal ID only account for
        // them, not the duplicates. We compute the lowest rank of the different merge that give
        // the same product.
        let mut lowest_rank: AHashMap<u32, u32> = AHashMap::new();
        for (rank, merge_id) in merges.values() {
            let slot = lowest_rank.entry(*merge_id).or_insert(*rank);
            *slot = cmp::min(*slot, *rank);
        }
        // (rank, product) packed rank-high into one u64. Ascending u64 order is exactly the
        // lexicographic order the `(u32, u32)` tuple sorted by, and it reuses the `[u64]` sort
        // above instead of monomorphising a second one for the tuple.
        let mut products: Vec<u64> = lowest_rank
            .iter()
            .map(|(p, r)| ((*r as u64) << 32) | *p as u64)
            .collect();
        products.sort_unstable();

        // this one is destroyed afterwards, does not matter if its big.
        let mut internal_id_map =
            vec![u32::MAX; *vocab.values().max().unwrap_or(&0u32) as usize + 1];
        let mut unmap = vec![u32::MAX; base + products.len()];
        // fill the first 0->base with the alphabet sorted by rank.
        for (internal, external) in alphabet.iter().enumerate() {
            let external = *external as u32;
            unmap[internal] = external;
            internal_id_map[external as usize] = internal as u32;
        }
        // now fill the rest of the tables with products sorted by rank.
        for (pos, packed) in products.iter().enumerate() {
            let internal = (base + pos) as u32;
            let product = *packed as u32;
            unmap[internal as usize] = product;
            internal_id_map[product as usize] = internal;
        }
        let (fold, byte_internal) =
            fold::build(&vocab, &merges, &internal_id_map, &unmap, byte_level);

        // For the SAFE flag: the cheapest rank at which a token appears as the left member of some
        // merge, and as the right member. A merge is safe to batch when its product cannot reach a
        // cheaper merge than the one being applied, on either side.
        let mut min_rank_left = vec![u32::MAX; unmap.len()];
        let mut min_rank_right = vec![u32::MAX; unmap.len()];
        for ((a, b), (rank, _)) in merges.iter() {
            if let Some(&ia) = internal_id_map.get(*a as usize)
                && (ia as usize) < min_rank_left.len()
            {
                min_rank_left[ia as usize] = min_rank_left[ia as usize].min(*rank);
            }
            if let Some(&ib) = internal_id_map.get(*b as usize)
                && (ib as usize) < min_rank_right.len()
            {
                min_rank_right[ib as usize] = min_rank_right[ib as usize].min(*rank);
            }
        }

        let mut top_merges = vec![u64::MAX; 512 * 512];
        let mut values = Vec::new();
        let mut keys = Vec::new();
        let mut dropped = 0usize;
        let mut unsafe_merges = 0usize;
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
            assert!(
                internal <= ID_MASK,
                "product id {internal} overflows the 30-bit id field"
            );
            let safe =
                *rank < min_rank_left[internal as usize].min(min_rank_right[internal as usize]);
            unsafe_merges += usize::from(!safe);
            let value = (*rank as u64) << 32 | if safe { SAFE_MASK } else { 0 } | internal;
            // if a and b < 512 -> Dense grid
            if (ia | ib) < 512 {
                top_merges[(ia << 9 | ib) as usize] = value;
            } else {
                keys.push((ia, ib));
                values.push(value);
            }
        }
        let unmap = unmap.into_boxed_slice();
        // compact: cells keep a u16 index into the live values
        let live = 512 * 512 - top_merges.iter().filter(|c| **c == u64::MAX).count();
        assert!(
            live < u16::MAX as usize,
            "{live} live grid entries exceed a u16 index; widen top_index to u32"
        );
        let mut top_index = vec![u16::MAX; 512 * 512];
        let mut top_values = Vec::with_capacity(live);
        for (slot, &value) in top_merges.iter().enumerate() {
            if value != u64::MAX {
                top_index[slot] = top_values.len() as u16;
                top_values.push(value);
            }
        }
        drop(top_merges);
        let top_index = top_index.into_boxed_slice();
        let top_values = top_values.into_boxed_slice();
        let pair_table = MphfMap::build(keys, values);
        info!(
            "bpe tables: {base} alphabet + {} products (unique merges), {} merge in the dense grid, {dropped} merges dropped, {unsafe_merges} merges unsafe to batch",
            products.len(),
            top_values.len()
        );
        (
            Self {
                unmap,
                pair_table,
                top_index,
                top_values,
                fold,
                byte_internal,
                any_unsafe: unsafe_merges > 0,
            },
            internal_id_map,
        )
    }
    /// Every merge these tables hold, as `(rank, left internal id, right internal id)`, in rank
    /// order.
    ///
    /// The inverse of the loop in [`Self::build`], and it is exact rather than a reconstruction:
    /// the dense grid encodes the pair in the slot index (`ia << 9 | ib`) and the perfect-hash map
    /// stores each key beside its value, so both halves of the split can be walked. Ranks are the
    /// positions the merge list was read in, so sorting by rank returns that list.
    ///
    /// Merges the build dropped -- a pair naming a token outside the vocabulary -- stay dropped,
    /// and a pair repeated in the source collapsed to one entry on the way in. Neither can change
    /// what the rebuilt tables do.
    pub(crate) fn merge_list(&self) -> Vec<(u32, u32, u32)> {
        let mut out = Vec::with_capacity(self.top_values.len() + self.pair_table.iter().count());
        for (slot, &index) in self.top_index.iter().enumerate() {
            if index != u16::MAX {
                let value = self.top_values[index as usize];
                out.push((
                    (value >> 32) as u32,
                    (slot >> 9) as u32,
                    (slot & 511) as u32,
                ));
            }
        }
        for (key, value) in self.pair_table.iter() {
            out.push(((value >> 32) as u32, (key >> 32) as u32, key as u32));
        }
        out.sort_unstable();
        out
    }

    /// `internal id -> external vocabulary id`.
    pub(crate) fn external(&self, internal: u32) -> Option<u32> {
        self.unmap.get(internal as usize).copied()
    }

    #[inline(always)]
    pub fn get_value(&self, a: &u32, b: &u32) -> u64 {
        if (a | b) < 512 {
            let slot = self.top_index.at((a << 9 | b) as usize);
            if slot == u16::MAX {
                u64::MAX
            } else {
                self.top_values.at(slot as usize)
            }
        } else {
            self.pair_table.get(((*a as u64) << 32) | *b as u64)
        }
    }
}

#[cfg(test)]
mod test {
    use ahash::AHashMap;

    use crate::models::bpe::{
        MergeMap,
        tables::{BpeTables, SAFE_MASK},
    };

    #[test]
    fn test_build() {
        let vocab = AHashMap::from_iter(vec![
            ("a".to_string(), 0),
            ("b".to_string(), 1),
            ("ab".to_string(), 2),
            ("aba".to_string(), 3),
        ]);
        let mut merges = MergeMap::new();
        merges.insert((0, 1), (0, 2));
        merges.insert((3, 0), (1, 3));
        let (tables, _) = BpeTables::build(vocab, merges, true);
        // there are only 4 elements because ab and aba are part of the vocab
        // so the alphabet is a,b and the ranks are ab and aba.
        // Both operands are < 512, so the merge lives in the dense grid, not the MPHF.
        // grid and pair table share the value layout, so both halves have to be right
        // (a, b) -> ab: rank 0, internal 2, and SAFE because `ab` is in no merge of its own, so
        // batching every occurrence of (a, b) in one sweep cannot skip a cheaper merge
        assert_eq!(tables.get_value(&0, &1), SAFE_MASK | 2);
        // (aba, a) -> aba: rank 1, internal 3, NOT safe: `aba` is the left member of that same
        // rank-1 merge, so the product can immediately form a pair no dearer than the one applied
        assert_eq!(tables.get_value(&3, &0), 1u64 << 32 | 3);
        assert!(tables.any_unsafe);
        assert_eq!(tables.get_value(&0, &2), u64::MAX); // (a, c) is not a merge
        assert_eq!(tables.pair_table.get(1u64), u64::MAX); // and nowhere else
        assert_eq!(&*tables.unmap, &[0, 1, 2, 3]);
    }
}
