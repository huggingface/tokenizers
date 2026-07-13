//! Thread-local pre-token → ids cache, L2-resident, MPHF-adjacent.
//!
//! Slots are open-addressed and store the pre-token's 64-bit hash plus its key
//! byte range and id range. On a hash hit we **verify the key bytes** before
//! returning the cached ids — perfect byte-exactness, no dependence on hash
//! uniqueness. (Set `FLATCACHE_HASHONLY=1` to skip the verify and measure its
//! cost; NOT byte-exact — a 64-bit collision would serve wrong ids.)
//!
//! Owned per thread via `thread_local!` → lock-free. Alloc-free in steady
//! state: fixed `Box<[CSlot]>`, key + id arenas pre-reserved and only cleared.

use ahash::RandomState;
use std::sync::OnceLock;

/// Skip the byte verify (hash-only). Read once. Default = verify (byte-exact).
fn hash_only() -> bool {
    static H: OnceLock<bool> = OnceLock::new();
    *H.get_or_init(|| std::env::var("FLATCACHE_HASHONLY").is_ok())
}

/// 24 bytes: hash(8) + koff(4) + ioff(4) + klen(2) + ilen(2). `klen == 0` = empty.
#[derive(Clone, Copy)]
struct CSlot {
    hash: u64,
    koff: u32,
    ioff: u32,
    klen: u16,
    ilen: u16,
}

pub(crate) struct FlatCache {
    slots: Box<[CSlot]>,
    kbytes: Vec<u8>,
    ids: Vec<u32>,
    mask: usize,
    count: usize,
    gen: u64,
    verify: bool,
    hasher: RandomState,
}

impl FlatCache {
    pub(crate) fn new() -> Self {
        let bits = std::env::var("CACHE_BITS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16usize);
        let slots_n = 1usize << bits;
        let slots = vec![CSlot { hash: 0, koff: 0, ioff: 0, klen: 0, ilen: 0 }; slots_n]
            .into_boxed_slice();
        FlatCache {
            slots,
            mask: slots_n - 1,
            count: 0,
            gen: u64::MAX,
            verify: !hash_only(),
            kbytes: Vec::with_capacity(slots_n * 16),
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
            s.klen = 0;
        }
        self.kbytes.clear();
        self.ids.clear();
        self.count = 0;
    }

    #[inline]
    pub(crate) fn retarget(&mut self, gen: u64) {
        if self.gen != gen {
            self.clear();
            self.gen = gen;
        }
    }

    /// Linear-probe lookup. On a hash match, verify the key bytes (unless
    /// hash-only); a byte mismatch keeps probing (the real entry may be later).
    #[inline]
    pub(crate) fn get(&self, p: &[u8], h: u64) -> Option<(u32, u16)> {
        let mut i = (h as usize) & self.mask;
        loop {
            let s = self.slots[i];
            if s.klen == 0 {
                return None; // empty slot
            }
            if s.hash == h
                && (!self.verify
                    || (s.klen as usize == p.len()
                        && self.kbytes[s.koff as usize..s.koff as usize + s.klen as usize] == *p))
            {
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
    pub(crate) fn insert(&mut self, p: &[u8], h: u64, ids: &[u32]) {
        if ids.is_empty()
            || p.len() > u16::MAX as usize
            || ids.len() > u16::MAX as usize
        {
            return;
        }
        if self.kbytes.len() + p.len() > self.kbytes.capacity()
            || self.ids.len() + ids.len() > self.ids.capacity()
            || self.count * 4 >= self.slots.len() * 3
        {
            self.clear();
        }
        let (koff, ioff) = (self.kbytes.len() as u32, self.ids.len() as u32);
        self.kbytes.extend_from_slice(p);
        self.ids.extend_from_slice(ids);
        let mut i = (h as usize) & self.mask;
        while self.slots[i].klen != 0 {
            i = (i + 1) & self.mask;
        }
        self.slots[i] = CSlot {
            hash: h,
            koff,
            ioff,
            klen: p.len() as u16,
            ilen: ids.len() as u16,
        };
        self.count += 1;
    }
}
