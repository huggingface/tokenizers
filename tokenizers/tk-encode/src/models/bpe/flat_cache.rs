//! Thread-local pre-token → ids cache, L2-resident, frequency-retaining.
//!
//! One pass over a text stream: a pre-token is cached on its first miss and
//! hits on every later occurrence (" the" pays off from its 2nd sighting). The
//! cache is sized to fit L2; when it fills we don't wipe it — we **cull**: keep
//! only entries that were actually reused (freq ≥ 1), halve their frequency
//! (aging), and drop the one-shots (freq 0 — the unique long pre-tokens that
//! merged once and never recur). So the hot Zipfian set survives; pollution
//! doesn't. Byte-exact (key verified on hit). Lock-free (thread-local).
//!
//! Toggles (measurement): CACHE_BITS (table size), CACHE_EVICT=clear (old
//! wipe-on-full), FLATCACHE_HASHONLY=1 (skip byte verify — not byte-exact).

use ahash::RandomState;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

// Process-wide hit/miss counters (CACHE_STATS=1 makes callers report them).
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);
pub fn flat_cache_stats() -> (u64, u64) {
    (HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed))
}
fn stats_on() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| std::env::var("CACHE_STATS").is_ok())
}
fn hash_only() -> bool {
    static H: OnceLock<bool> = OnceLock::new();
    *H.get_or_init(|| std::env::var("FLATCACHE_HASHONLY").is_ok())
}
fn clear_on_full() -> bool {
    static C: OnceLock<bool> = OnceLock::new();
    *C.get_or_init(|| std::env::var("CACHE_EVICT").map(|v| v == "clear").unwrap_or(false))
}

/// hash(8) + key(16) + koff(4) + ioff(4) + klen(2) + ilen(2) + freq(1). `key != 0` = a ≤15-byte
/// pre-token packed inline (length in top byte): a hit is a register 128-bit compare, no `kbytes`
/// load, no `memcmp`. `key == 0` = a long (>15B) pre-token, verified the old way via `kbytes`.
#[derive(Clone, Copy)]
struct CSlot {
    hash: u64,
    key: u128,
    koff: u32,
    ioff: u32,
    klen: u16,
    ilen: u16,
    freq: u8,
}
const EMPTY: CSlot = CSlot { hash: 0, key: 0, koff: 0, ioff: 0, klen: 0, ilen: 0, freq: 0 };

pub(crate) struct FlatCache {
    slots: Box<[CSlot]>,
    kbytes: Vec<u8>,
    ids: Vec<u32>,
    // second buffers for the compacting cull (swapped in, so the hot path stays alloc-free).
    slots2: Box<[CSlot]>,
    kbytes2: Vec<u8>,
    ids2: Vec<u32>,
    mask: usize,
    count: usize,
    gen: u64,
    verify: bool,
    clear_full: bool,
    stats: bool,
    hasher: RandomState,
}

impl FlatCache {
    pub(crate) fn new() -> Self {
        let bits = std::env::var("CACHE_BITS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16usize);
        let slots_n = 1usize << bits;
        FlatCache {
            slots: vec![EMPTY; slots_n].into_boxed_slice(),
            slots2: vec![EMPTY; slots_n].into_boxed_slice(),
            mask: slots_n - 1,
            count: 0,
            gen: u64::MAX,
            verify: !hash_only(),
            clear_full: clear_on_full(),
            stats: stats_on(),
            kbytes: Vec::with_capacity(slots_n * 16),
            ids: Vec::with_capacity(slots_n * 8),
            kbytes2: Vec::with_capacity(slots_n * 16),
            ids2: Vec::with_capacity(slots_n * 8),
            hasher: RandomState::with_seeds(0xC0FFEE, 0xBADF00D, 0xDEADBEEF, 0x1337),
        }
    }

    #[inline]
    pub(crate) fn hash(&self, p: &[u8]) -> u64 {
        use std::hash::BuildHasher;
        self.hasher.hash_one(p)
    }

    fn clear(&mut self) {
        for s in self.slots.iter_mut() {
            s.klen = 0;
        }
        self.kbytes.clear();
        self.ids.clear();
        self.count = 0;
    }

    /// Generational cull: compact into the scratch buffers, keeping only entries
    /// reused since the last cull (freq ≥ 1), halving freq (aging), dropping the
    /// freq-0 one-shots. Swap the buffers in. If almost everything survived (the
    /// working set genuinely exceeds the table), fall back to a clear so we make
    /// room instead of culling every insert.
    fn cull(&mut self) {
        if self.clear_full {
            self.clear();
            return;
        }
        let Self {
            slots, slots2, kbytes, kbytes2, ids, ids2, mask, ..
        } = self;
        for s in slots2.iter_mut() {
            s.klen = 0;
        }
        kbytes2.clear();
        ids2.clear();
        let mut kept = 0usize;
        for s in slots.iter() {
            if s.klen == 0 || s.freq == 0 {
                continue; // empty or one-shot → drop
            }
            let (ks, ke) = (s.koff as usize, s.koff as usize + s.klen as usize);
            let (is, ie) = (s.ioff as usize, s.ioff as usize + s.ilen as usize);
            let koff = kbytes2.len() as u32;
            let ioff = ids2.len() as u32;
            kbytes2.extend_from_slice(&kbytes[ks..ke]);
            ids2.extend_from_slice(&ids[is..ie]);
            let mut i = (s.hash as usize) & *mask;
            while slots2[i].klen != 0 {
                i = (i + 1) & *mask;
            }
            slots2[i] = CSlot {
                hash: s.hash,
                key: s.key,
                koff,
                ioff,
                klen: s.klen,
                ilen: s.ilen,
                freq: s.freq >> 1,
            };
            kept += 1;
        }
        std::mem::swap(slots, slots2);
        std::mem::swap(kbytes, kbytes2);
        std::mem::swap(ids, ids2);
        self.count = kept;
        // Everything was hot → culling freed nothing; clear to avoid culling every insert.
        if kept * 4 >= self.slots.len() * 3 {
            self.clear();
        }
    }

    #[inline]
    pub(crate) fn retarget(&mut self, gen: u64) {
        if self.gen != gen {
            self.clear();
            self.gen = gen;
        }
    }

    /// Lookup + bump the entry's frequency on a hit (so the cull can tell hot
    /// from one-shot). Needs `&mut self`.
    ///
    /// `key != 0`: a packed ≤15-byte pre-token — confirm the hit with a register 128-bit compare
    /// (`s.key == key`), no `kbytes` load, no `memcmp`. `key == 0`: long pre-token — byte-verify
    /// via `kbytes` as before (unless hash-only).
    #[inline]
    pub(crate) fn get(&mut self, p: &[u8], h: u64, key: u128) -> Option<(u32, u16)> {
        let mut i = (h as usize) & self.mask;
        loop {
            let s = self.slots[i];
            if s.klen == 0 {
                if self.stats {
                    MISSES.fetch_add(1, Ordering::Relaxed);
                }
                return None;
            }
            let confirmed = if key != 0 {
                s.key == key // register compare — no arena, no memcmp
            } else {
                s.hash == h
                    && (!self.verify
                        || (s.klen as usize == p.len()
                            && self.kbytes[s.koff as usize..s.koff as usize + s.klen as usize]
                                == *p))
            };
            if confirmed {
                self.slots[i].freq = self.slots[i].freq.saturating_add(1);
                if self.stats {
                    HITS.fetch_add(1, Ordering::Relaxed);
                }
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
    pub(crate) fn insert(&mut self, p: &[u8], h: u64, key: u128, ids: &[u32]) {
        if ids.is_empty() || p.len() > u16::MAX as usize || ids.len() > u16::MAX as usize {
            return;
        }
        if self.kbytes.len() + p.len() > self.kbytes.capacity()
            || self.ids.len() + ids.len() > self.ids.capacity()
            || self.count * 4 >= self.slots.len() * 3
        {
            self.cull();
        }
        let (koff, ioff) = (self.kbytes.len() as u32, self.ids.len() as u32);
        // kbytes retained for all (cull compacts by koff/klen); packed-key gets never read it.
        self.kbytes.extend_from_slice(p);
        self.ids.extend_from_slice(ids);
        let mut i = (h as usize) & self.mask;
        while self.slots[i].klen != 0 {
            i = (i + 1) & self.mask;
        }
        self.slots[i] = CSlot {
            hash: h,
            key,
            koff,
            ioff,
            klen: p.len() as u16,
            ilen: ids.len() as u16,
            freq: 0, // on probation until reused
        };
        self.count += 1;
    }
}
