//! Thread-local pre-token → ids cache, L2-resident, hash-keyed.
//!
//! Design (per the "keep the fixed cost tiny" goal):
//!  * **Hash-only slots** — a slot stores the 64-bit pre-token hash and the
//!    id-list range, NOT the key bytes. Lookup is a pure integer compare; there
//!    is no key arena and no byte walk. Trade-off: a 64-bit hash collision would
//!    serve wrong ids (~2^-50 per lookup at these table sizes; see below) — so
//!    this is *not* a hard byte-exact guarantee. A production build would widen
//!    to 128-bit or verify-on-match.
//!  * **Small enough for L2** — default 2^14 slots × 16 B = 256 KB of slots plus
//!    a small id arena, so the hot table stays resident. Tunable via CACHE_BITS.
//!  * **Only long pre-tokens are ever inserted** (the caller applies a minimum
//!    length) — short ones re-merge for ~free, so they'd only pollute the table.
//!
//! Owned per thread via `thread_local!` → lock-free. Alloc-free in steady state:
//! `slots` is a fixed `Box<[_]>`, the id arena is pre-reserved and only cleared.

use ahash::RandomState;

/// 16 bytes: 8 (hash) + 4 (ioff) + 2 (ilen) + 2 pad. `ilen == 0` marks empty.
#[derive(Clone, Copy)]
struct CSlot {
    hash: u64,
    ioff: u32,
    ilen: u16,
}

pub(crate) struct FlatCache {
    slots: Box<[CSlot]>,
    ids: Vec<u32>,
    mask: usize,
    count: usize,
    gen: u64,
    hasher: RandomState,
}

impl FlatCache {
    pub(crate) fn new() -> Self {
        // Default 2^14 slots (256 KB of slots) so the table fits L2.
        // Default 2^16 slots: the sweep showed clear-on-full thrashes when the
        // table is smaller than the corpus's distinct-pretoken set, so bigger
        // wins until a real LRU lands. Tunable via CACHE_BITS.
        let bits = std::env::var("CACHE_BITS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16usize);
        let slots_n = 1usize << bits;
        let slots = vec![CSlot { hash: 0, ioff: 0, ilen: 0 }; slots_n].into_boxed_slice();
        FlatCache {
            slots,
            mask: slots_n - 1,
            count: 0,
            gen: u64::MAX,
            // ~8 ids per cached (long) pre-token, pre-reserved.
            ids: Vec::with_capacity(slots_n * 8),
            hasher: RandomState::with_seeds(0xC0FFEE, 0xBADF00D, 0xDEADBEEF, 0x1337),
        }
    }

    #[inline]
    pub(crate) fn hash(&self, p: &[u8]) -> u64 {
        use std::hash::BuildHasher;
        self.hasher.hash_one(p)
    }

    #[inline]
    fn clear(&mut self) {
        for s in self.slots.iter_mut() {
            s.ilen = 0;
        }
        self.ids.clear();
        self.count = 0;
    }

    /// Point the cache at a given BPE instance; clear if it changed.
    #[inline]
    pub(crate) fn retarget(&mut self, gen: u64) {
        if self.gen != gen {
            self.clear();
            self.gen = gen;
        }
    }

    /// Hash-only lookup: linear probe, integer compare, no key bytes.
    #[inline]
    pub(crate) fn get(&self, h: u64) -> Option<(u32, u16)> {
        let mut i = (h as usize) & self.mask;
        loop {
            let s = self.slots[i];
            if s.ilen == 0 {
                return None; // empty slot
            }
            if s.hash == h {
                return Some((s.ioff, s.ilen));
            }
            i = (i + 1) & self.mask;
        }
    }

    #[inline]
    pub(crate) fn ids_slice(&self, off: u32, len: u16) -> &[u32] {
        &self.ids[off as usize..off as usize + len as usize]
    }

    #[inline]
    pub(crate) fn insert(&mut self, h: u64, ids: &[u32]) {
        if ids.is_empty() || ids.len() > u16::MAX as usize {
            return;
        }
        if self.ids.len() + ids.len() > self.ids.capacity() || self.count * 4 >= self.slots.len() * 3
        {
            self.clear();
        }
        let ioff = self.ids.len() as u32;
        self.ids.extend_from_slice(ids);
        let mut i = (h as usize) & self.mask;
        while self.slots[i].ilen != 0 {
            i = (i + 1) & self.mask;
        }
        self.slots[i] = CSlot {
            hash: h,
            ioff,
            ilen: ids.len() as u16,
        };
        self.count += 1;
    }
}
