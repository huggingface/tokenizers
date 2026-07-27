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
    pair_table: Box<[u64]>,
    top_merges: [u32; 1024 * 1024],
    merge_rank_left: Box<[bool]>,
    merge_rank_right: Box<[bool]>,
    fold: [u32; 65536],
}

impl BpeTables {
    pub(crate) fn build(vocab: impl IntoIterator, merges: impl IntoIterator) -> Self {
        // 1. We build the internal id map

        todo!()
    }
}
