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

// The packed pair key is fed to ptr_hash directly: `index_single_part` already
// hashes the u64 internally (FxHash — it showed up in the profile), so an extra
// splitmix pass here was pure waste. Packed pairs are distinct by construction,
// so no dedup is needed.

#[repr(align(16))]
#[derive(Clone, Copy)]
struct Entry {
    key: u64,
    rank: u32,
    merged: u32,
}

/// 8-bit fingerprint of a packed key, independent of the MPHF's internal hash.
#[inline(always)]
fn fp_of(key: u64) -> u8 {
    (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 56) as u8
}

pub(crate) struct RankStore {
    mphf: Mphf,
    entries: Box<[Entry]>,
    // #2 fingerprint pre-filter: fp[slot] == fp_of(key-at-slot). A query whose
    // fingerprint misses rejects the (absent) pair without the wide entry load.
    fp: Box<[u8]>,
    // #3 direct table for byte-byte pairs (a,b < 256): (rank<<32)|merged, or
    // u64::MAX for "no merge". No hash, no verify — direct index is exact.
    small: Box<[u64]>,
    use_fp: bool,
    use_small: bool,
}

const SMALL_N: usize = 1 << 16; // (a<<8)|b for a,b < 256

impl RankStore {
    pub(crate) fn build(merges: &MergeMap) -> Self {
        let mut params = PtrHashParams::default_fast();
        params.single_part = true;
        // #3 small direct table: default ON (benched +3..16%); RANK_NO_SMALL opts out.
        // #2 fingerprint: opt-in only (benched neutral-to-negative).
        let use_fp = std::env::var("RANK_FP").is_ok();
        let use_small = std::env::var("RANK_NO_SMALL").is_err();
        if merges.is_empty() {
            let empty: [u64; 0] = [];
            return RankStore {
                mphf: Mphf::new(&empty, params),
                entries: Box::new([]),
                fp: Box::new([]),
                small: Box::new([]),
                use_fp,
                use_small,
            };
        }
        // Collect once so build order is stable; each merged id is unique so
        // every slot is written exactly once (image gotcha #5).
        let pairs: Vec<((u32, u32), (u32, u32))> =
            merges.iter().map(|(&p, &v)| (p, v)).collect();
        let keys: Vec<u64> = pairs.iter().map(|&((a, b), _)| pack(a, b)).collect();
        let mphf = Mphf::new(&keys, params);
        let n = pairs.len();
        let mut entries = vec![
            Entry { key: u64::MAX, rank: 0, merged: 0 };
            n
        ]
        .into_boxed_slice();
        for &((a, b), (rank, merged)) in &pairs {
            let k = pack(a, b);
            let slot = mphf.index_single_part(&k);
            entries[slot] = Entry { key: k, rank, merged };
        }
        debug_assert!(
            entries.iter().all(|e| e.key != u64::MAX),
            "RankStore mis-sized: a slot was never written"
        );
        // #2 fingerprint per slot — only allocated when enabled.
        let fp: Box<[u8]> = if use_fp {
            entries.iter().map(|e| fp_of(e.key)).collect()
        } else {
            Box::new([])
        };
        // #3 direct byte-byte table — only allocated when enabled (default on).
        let small = if use_small {
            let mut s = vec![u64::MAX; SMALL_N].into_boxed_slice();
            for &((a, b), (rank, merged)) in &pairs {
                if a < 256 && b < 256 {
                    s[((a << 8) | b) as usize] = ((rank as u64) << 32) | merged as u64;
                }
            }
            s
        } else {
            Box::new([])
        };
        RankStore { mphf, entries, fp, small, use_fp, use_small }
    }

    /// One MPHF compute + one aligned `Entry` read + key verify. `None` for
    /// out-of-vocab pairs (the MPHF returns an arbitrary slot for those; the
    /// key check is what rejects them — not optional).
    #[cfg_attr(not(feature = "profile-noinline"), inline)]
    #[cfg_attr(feature = "profile-noinline", inline(never))]
    pub(crate) fn get(&self, a: u32, b: u32) -> Option<(u32, u32)> {
        if self.entries.is_empty() {
            return None;
        }
        // #3 byte-byte pairs: direct index, no hash, no verify.
        if self.use_small && a < 256 && b < 256 {
            let v = self.small[((a << 8) | b) as usize];
            return if v == u64::MAX {
                None
            } else {
                Some(((v >> 32) as u32, v as u32))
            };
        }
        let key = pack(a, b);
        let slot = self.mphf.index_single_part(&key);
        // #2 fingerprint reject: skips the wide entry load for absent pairs.
        if self.use_fp && self.fp[slot] != fp_of(key) {
            return None;
        }
        let e = self.entries[slot];
        if e.key == key {
            Some((e.rank, e.merged))
        } else {
            None
        }
    }
}
