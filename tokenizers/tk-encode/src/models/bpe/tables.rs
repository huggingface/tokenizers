use ahash::RandomState;
use ahash::{AHashMap, HashSet};
use ptr_hash::{FastPtrHash, PtrHashParams, hash::NoHash};
use std::cmp;

type Mphf = FastPtrHash<NoHash, u64>;

use crate::models::bpe::MergeMap;
use crate::models::bpe::bytelevel_folding::{ByteLevelFold, Fold};

/// Pair-table value layout: `rank[63:32] | flags[31:30] | internal_id[29:0]`, sentinel `u64::MAX`.
/// Rank is shifted to the high half so `val < min_val` is a rank comparison without having to do any
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
              // rank sits high so `val < min_val` is a rank comparison. Bit 30 is SAFE; bit 31 is
              // free. mrl/mrr are NOT stored here: they are build-time only, consumed by the fold
              // guard and by SAFE.
}

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
    #[inline]
    // from the key pair, returns the rank, the flags and the new id.
    pub fn get(&self, key: u64) -> u64 {
        let slot = self.mphf.index(&self.hasher.hash_one(key));
        let e = &self.entries[slot];
        if e.key == key { e.val } else { u64::MAX }
    }
}
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

/// NOTE: Unchecked indexing, justified once instead of everywhere we do it.
///
/// Every use of `.at()` in the fold and conversion paths is safe: the
/// bytes come from a `&str`, so a sequence length taken from a lead byte cannot run past the end;
/// and every table index is masked to that table's fixed size (`& 0x0F` << 6 | `& 0x3F` <= 1023,
/// `& 0x3F` < 64, a `u8` into `[_; 256]`). It exists because bounds-checked indexing measured
/// 25-44% slower on conversion.
pub trait At {
    type Out;
    fn at(&self, index: usize) -> Self::Out;
}

impl<T: Copy> At for [T] {
    type Out = T;
    #[inline(always)]
    fn at(&self, index: usize) -> T {
        unsafe { *self.get_unchecked(index) }
    }
}

/// A bitmap of which codepoints fold, plus the symbols they fold to.
///
/// A codepoint fits in a u16, so there are 65536 of them and one bit each is 65536 bits = 1024
/// u64s = 8 KB. `rows` is that Vec of u64. Splitting a codepoint into a row and a column is just
/// dividing by 64 and taking the remainder, and 64 is a power of two, so it is a shift and a mask:
///
///   row = codepoint >> 6        col = codepoint & 0x3F
///
///   rows:  [    u64    |    u64    |    u64    | ... |    u64    ]   1024 rows, 8 KB
///            row 0       row 1       row 2             row 1023
///            cp 0..63     cp 64..127
///
///   one row is 64 codepoints, one bit each:
///
///   row 192:   bit 63 <-------------------------------------- bit 0
///              0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0
///                    ^                       ^
///                    col 60 folds            col 16 folds
///
/// A set bit means that codepoint folds to one symbol. `symbols` holds those symbols and nothing
/// else, packed in codepoint order: 13 KB instead of 256 KB for a flat `[u32; 65536]`
/// codepoint's slot in the `symbols` table, count the set bits before it: `row_start` has the count for all
/// earlier rows, and one popcount covers the bits before `col` in its own row.
pub struct SparseFold {
    /// One bit per codepoint: set iff it folds. If it does we emit the corresponding u32 directly.
    rows: Box<[u64]>,
    /// indexed by the row, indexes the symbols
    row_start: Box<[u32]>,
    /// The symbols of folding codepoints only, in codepoint order.
    symbols: Box<[u32]>,
    /// One-byte characters, which are always exactly one symbol whether they fold or not. 512 B.
    ascii: [u32; 128],
    /// The same mapping for codepoints past 0xFFFF (emoji, CJK ext). Too few and too spread out to
    /// be worth optimizing at all.
    non_bmp: AHashMap<char, u32>,
}

impl SparseFold {
    fn build(
        codepoint_to_symbol: &[u32],
        byte_symbols: &[u32; 256],
        non_bmp: AHashMap<char, u32>,
    ) -> Self {
        let mut rows = vec![0u64; 1024];
        for (codepoint, &symbol) in codepoint_to_symbol.iter().enumerate() {
            if symbol != u32::MAX {
                // we set a single bit using |
                rows[codepoint >> 6] |= 1u64 << (codepoint & 0x3F);
            }
        }
        let mut row_start = vec![0u32; 1024];
        let mut seen = 0u32;
        for row in 0..1024 {
            row_start[row] = seen;
            seen += rows[row].count_ones();
        }
        let symbols: Vec<u32> = codepoint_to_symbol
            .iter()
            .copied()
            .filter(|&symbol| symbol != u32::MAX)
            .collect();
        let mut ascii = [0u32; 128];
        for (byte, symbol) in ascii.iter_mut().enumerate() {
            *symbol = if codepoint_to_symbol[byte] != u32::MAX {
                codepoint_to_symbol[byte]
            } else {
                byte_symbols[byte]
            };
        }
        Self {
            rows: rows.into_boxed_slice(),
            row_start: row_start.into_boxed_slice(),
            symbols: symbols.into_boxed_slice(),
            ascii,
            non_bmp,
        }
    }

    pub fn footprint(&self) -> usize {
        self.rows.len() * 8 + self.row_start.len() * 4 + self.symbols.len() * 4 + 512
    }

    /// The symbol at (row, col), or `u32::MAX` if that codepoint does not fold.
    #[inline(always)]
    fn get(&self, row: usize, col: u32) -> u32 {
        let bits = self.rows.at(row);
        if (bits >> col) & 1 == 0 {
            return u32::MAX;
        }
        let before =
            self.row_start.at(row) as usize + (bits & ((1u64 << col) - 1)).count_ones() as usize;
        self.symbols.at(before)
    }

    /// A one-byte character. Always one symbol, fold or not.
    #[inline(always)]
    pub fn get_ascii(&self, byte: u8) -> u32 {
        self.ascii.at((byte & 0x7F) as usize)
    }

    /// A character given as UTF-8 bytes. `u32::MAX` means it does not fold and the caller emits its
    /// bytes instead. `lead` and `char_len` come from the caller, which already has them.
    ///
    /// We use a small trick: any continuation byte can be converted to a key ton index or sparse fold.
    /// For:
    ///   3 bytes:  1110xxxx 10yyyyyy 10zzzzzz     row = xxxx yyyyyy     col = zzzzzz
    ///             (0F)1111   111111(3F)
    ///
    ///   2 bytes:  110yyyyy 10zzzzzz              row =      yyyyy      col = zzzzzz
    ///            (1F)11111   111111(3F)
    #[inline(always)]
    pub fn get_bytes(&self, bytes: &[u8], start: usize, lead: u8, char_len: usize) -> u32 {
        match char_len {
            3 => self.get(
                (((lead & 0x0F) as usize) << 6) | (bytes.at(start + 1) & 0x3F) as usize,
                (bytes.at(start + 2) & 0x3F) as u32,
            ),
            2 => self.get((lead & 0x1F) as usize, (bytes.at(start + 1) & 0x3F) as u32),
            // four bytes: past the BitMapPlane, so the bitmap does not cover it
            _ => {
                let codepoint = (((lead & 0x07) as u32) << 18)
                    | (((bytes.at(start + 1) & 0x3F) as u32) << 12)
                    | (((bytes.at(start + 2) & 0x3F) as u32) << 6)
                    | (bytes.at(start + 3) & 0x3F) as u32;
                self.get_code(codepoint)
            }
        }
    }

    /// A character, for models whose atoms are characters rather than bytes.
    #[inline(always)]
    pub fn get_char(&self, character: char) -> u32 {
        self.get_code(character as u32)
    }

    #[inline(always)]
    fn get_code(&self, codepoint: u32) -> u32 {
        if codepoint < 0x10000 {
            // in that case we already have the codepoint so we need less masking than utf8.
            self.get(codepoint as usize >> 6, codepoint & 0x3F)
        } else {
            char::from_u32(codepoint)
                .and_then(|character| self.non_bmp.get(&character).copied())
                .unwrap_or(u32::MAX)
        }
    }
}

/// UTF-8 sequence length by lead byte.
pub const UTF8_LEN: [u8; 256] = {
    let mut l = [1u8; 256];
    let mut b = 0xC0usize;
    while b < 0xE0 {
        l[b] = 2;
        b += 1;
    }
    while b < 0xF0 {
        l[b] = 3;
        b += 1;
    }
    while b < 0xF8 {
        l[b] = 4;
        b += 1;
    }
    l
};

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
        for (rank, merge_id) in merges.values() {
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
        let (cp_to_internal_id, non_bmp, byte_internal) =
            build_conversion_table(&vocab, &merges, &internal_id_map, &unmap, byte_level);
        // the flat 256 KB table is build-time only: it is compacted here and dropped
        let fold = SparseFold::build(&cp_to_internal_id, &byte_internal, non_bmp);
        drop(cp_to_internal_id);
        info!(
            "fold table: {:.1} KB sparse (flat would be {:.1} KB)",
            fold.footprint() as f64 / 1024.0,
            65536.0 * 4.0 / 1024.0
        );

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

/// We build the codepoint character to internal id table.
fn build_conversion_table(
    vocab: &AHashMap<String, u32>,
    merges: &MergeMap,
    internal_id_map: &[u32],
    unmap: &[u32],
    byte_level: bool,
) -> (Vec<u32>, AHashMap<char, u32>, [u32; 256]) {
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
    let mut byte_internal = [u32::MAX; 256];
    if byte_level {
        // A character reaches the merge loop as bytes, so folding it means proving the
        // merges are  predetermined. See `bytelevel_folding`.
        let folder = ByteLevelFold::new(vocab, merges, internal_id_map, unmap);
        byte_internal = folder.byte_internal();
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
    (cp_to_internal_id, non_bmp, byte_internal)
}

#[cfg(test)]
mod test {
    use ahash::AHashMap;

    use crate::models::bpe::{
        MergeMap,
        tables::{BpeTables, MphfMap, SAFE_MASK},
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
        let value = 1u64 << 32 | 5_u64;
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
