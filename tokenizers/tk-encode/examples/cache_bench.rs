//! Head-to-head benchmark of the BPE word-cache designs this branch went
//! through, on the real corpora: for every BPE model in
//! `examples/bench_models.json` and every fixture in `data/fixtures/`, replays
//! the exact word stream the pipeline model sees through each cache variant
//! (misses pay the real merge cost), sweeping capacity × max key length, and
//! times the shipped pipeline end-to-end with the cache on vs off. Emits one
//! JSON object on stdout.
//!
//!     cargo run --release --features fancy-regex --example cache_bench
//!     cargo run --release --features fancy-regex --example cache_bench -- \
//!         --models gpt2,llama-3 --fixtures eng,kor \
//!         --capacities 10000,65536 --max-lengths 64,256 \
//!         --reps 3 --out cache_bench.json
//!
//! Models whose tokenizer needs a system regex engine (llama-2) are skipped
//! unless built with `--features fancy-regex`.
//!
//! Variants, all driven through the same replay loop so any difference is the
//! data structure alone:
//! - `naive_hashmap` — this branch's v1: `AHashMap<String, Box<[u32]>>`,
//!   arbitrary evict-one at capacity. Allocates key + value on every insert.
//!   (v1 had no key-length guard; here it gets the same `max_length` as the
//!   others so the sweep stays comparable.)
//! - `direct_mapped` — fixed slot table + flat key/id arenas, 1 slot per hash.
//! - `assoc_4way` — the shipped design: replica of
//!   `src/models/bpe/word_cache.rs` (4 × 16 B slots = one 64 B cache line).
//! - `assoc_8way` — same with 8-slot buckets (two cache lines): the
//!   "why not more ways" control.
//! - `flat_cache` — replica of PR #2234's `FlatCache` (perf/fused-split-cache):
//!   open-addressed linear probing at <= 75% load, <= 15-byte keys packed into a
//!   u128 and compared in a register (CRC hash), a frequency byte per slot, and
//!   a generational cull on fill (keep reused entries, halve their frequency,
//!   drop one-shots). No key-length cap by design, so it runs once per capacity
//!   and reports `max_length: null`. This measures the cache structure under
//!   the shared replay loop — not #2234's fused-split/de-virtualized pipeline.
//! - `bucket_cull` — the hybrid: assoc_4way's 16 B slots and one-line buckets,
//!   plus flat_cache's retention idea. A reuse flag in a spare `key_len` bit,
//!   CLOCK-style second-chance eviction inside the bucket (rejecting an insert
//!   ages the bucket), arena compaction on a budget trip.
//!
//! Per (model × fixture × capacity × max_length × variant):
//! - warm replay throughput and speedup vs the no-cache floor
//! - steady-state hit/miss rate
//! - insert outcomes: stored / evicted / rejected (bucket full, key too long)
//! - collision rate on insert (stored into a bucket already holding entries;
//!   not observable for the hashmap) and false tag matches (16-slot probe
//!   matched the 32-bit tag but the key compare failed — hash quality)
//! - occupancy and memory footprint (table / keys / ids breakdown)
//! - per-op latency percentiles from dedicated instrumented passes: hit and
//!   miss at steady state, miss on a cold table, insert. Raw wall-clock per
//!   op; `meta.timer` gives the measured clock overhead and resolution to
//!   read them against. Throughput passes are never per-op timed.
//!
//! Fidelity caveats, also embedded in the JSON `meta.notes`:
//! - The fixed-table variants are hand-kept replicas of `word_cache.rs`; the
//!   end-to-end cache-on run exercises the real shipped code and cross-checks
//!   them. End-to-end sweeps capacity only — the shipped `MAX_LENGTH` is a
//!   crate constant.
//! - Replay probes the cache before the model's `ignore_merges` whole-word
//!   vocab check; the shipped cache sits after it. On `ignore_merges` models
//!   (llama-3 family) replay caches whole-vocab words the shipped cache never
//!   sees — variant-vs-variant stays fair, absolute hit rates read high.
//! - The shipped cache seeds its hasher randomly per process, so its conflict
//!   set (not its design) varies run to run; the replicas use fixed seeds.
//! - Every replayed id sequence is checked against `PipelineTokenizer::encode`
//!   and the result reported as `ids_match`. Chunks the pipeline encodes
//!   through added-token segmentation (which replay lacks) are dropped up
//!   front and counted in `chunks_skipped`.

use std::borrow::Cow;
use std::convert::{TryFrom, TryInto};
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use ahash::{AHashMap, AHashSet, RandomState};
use serde_json::{Value, json};
use tk_encode::pipeline::{
    Model, Normalizer, PipelineModel, PipelinePreTokenizer, PipelineToken, PreTokenizer, Span,
};
use tk_encode::{ModelWrapper, Tokenizer, pipeline::PipelineTokenizer};

const DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data");
const MANIFEST: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/bench_models.json");
const CHUNK_BYTES: usize = 10 * 1024;
const MAX_CHUNKS: usize = 100;
const PROBE: &str = "The quick brown fox jumps 123.";
// Shipped values, `utils/cache.rs` (not public): DEFAULT_CACHE_CAPACITY and
// MAX_LENGTH. Keep in sync.
const DEFAULT_CAPACITIES: &[usize] = &[10_000, 65_536, 262_144];
const DEFAULT_MAX_LENGTHS: &[usize] = &[64, 256, 1024];
const SEEDS: [u64; 4] = [0x0123, 0x4567, 0x89ab, 0xcdef];

fn fixed_hasher() -> RandomState {
    RandomState::with_seeds(SEEDS[0], SEEDS[1], SEEDS[2], SEEDS[3])
}

// ── cache variants ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum InsertOutcome {
    Stored,
    /// Stored, displacing an arbitrary existing entry (hashmap at capacity).
    Evicted,
    RejectedFull,
    RejectedLen,
}

struct MemBreakdown {
    table: usize,
    keys: usize,
    ids: usize,
}

trait CacheVariant {
    /// `&mut` because some designs touch state on lookup (flat_cache bumps a
    /// frequency counter on hit).
    fn get(&mut self, word: &str) -> Option<&[u32]>;
    fn insert(&mut self, word: &str, ids: &[u32]) -> InsertOutcome;
    /// Generational culls performed, for designs that make room in batches.
    fn culls(&self) -> Option<usize> {
        None
    }
    /// Occupied slots in the bucket `word` hashes to, after any insert.
    /// None when the structure has no observable buckets (hashmap).
    fn bucket_load(&self, word: &str) -> Option<usize>;
    /// Slots in `word`'s bucket whose tag matches — a superset of a real hit.
    /// The excess over hits measures 32-bit tag false positives.
    fn probe_tags(&self, word: &str) -> Option<u32>;
    fn occupied(&self) -> usize;
    fn slot_count(&self) -> usize;
    fn memory(&self) -> MemBreakdown;
}

/// No cache: the merge-loop floor every variant is compared against.
struct NoCache;

impl CacheVariant for NoCache {
    fn get(&mut self, _: &str) -> Option<&[u32]> {
        None
    }
    fn insert(&mut self, _: &str, _: &[u32]) -> InsertOutcome {
        InsertOutcome::RejectedFull
    }
    fn bucket_load(&self, _: &str) -> Option<usize> {
        None
    }
    fn probe_tags(&self, _: &str) -> Option<u32> {
        None
    }
    fn occupied(&self) -> usize {
        0
    }
    fn slot_count(&self) -> usize {
        0
    }
    fn memory(&self) -> MemBreakdown {
        MemBreakdown { table: 0, keys: 0, ids: 0 }
    }
}

/// This branch's v1 (commit b5ef46fe): owned `String` keys, a boxed id slice
/// per entry, arbitrary evict-one at capacity.
struct NaiveCache {
    capacity: usize,
    max_length: usize,
    lookup: AHashMap<String, Box<[u32]>>,
}

impl NaiveCache {
    fn new(capacity: usize, max_length: usize) -> Self {
        Self {
            capacity,
            max_length,
            lookup: AHashMap::with_capacity_and_hasher(capacity, fixed_hasher()),
        }
    }
}

impl CacheVariant for NaiveCache {
    fn get(&mut self, word: &str) -> Option<&[u32]> {
        if word.len() > self.max_length {
            return None;
        }
        self.lookup.get(word).map(|bx| &bx[..])
    }

    fn insert(&mut self, word: &str, ids: &[u32]) -> InsertOutcome {
        if word.len() > self.max_length {
            return InsertOutcome::RejectedLen;
        }
        let evict = self.lookup.len() >= self.capacity;
        if evict {
            self.lookup.extract_if(|_, _| true).next();
        }
        self.lookup
            .insert(word.to_owned(), ids.to_vec().into_boxed_slice());
        if evict { InsertOutcome::Evicted } else { InsertOutcome::Stored }
    }

    fn bucket_load(&self, _: &str) -> Option<usize> {
        None
    }
    fn probe_tags(&self, _: &str) -> Option<u32> {
        None
    }
    fn occupied(&self) -> usize {
        self.lookup.len()
    }
    fn slot_count(&self) -> usize {
        self.capacity
    }
    /// Table is an estimate: hashmap buckets + 1 control byte per slot.
    /// Ignores allocator slack.
    fn memory(&self) -> MemBreakdown {
        MemBreakdown {
            table: self.lookup.capacity()
                * (size_of::<String>() + size_of::<Box<[u32]>>() + 1),
            keys: self.lookup.keys().map(String::len).sum(),
            ids: self.lookup.values().map(|v| v.len() * size_of::<u32>()).sum(),
        }
    }
}

// Replica of `src/models/bpe/word_cache.rs`, generic over the bucket so the
// same probe/insert code runs as 1-, 4-, and 8-way. Keep in sync by hand.

#[derive(Clone, Copy, Default)]
struct CacheSlot {
    tag: u32,
    key_off: u32,
    ids_off: u32,
    key_len: u16,
    ids_len: u16,
}

impl CacheSlot {
    fn id_range(&self) -> std::ops::Range<usize> {
        self.ids_off as usize..(self.ids_off as usize + self.ids_len as usize)
    }

    fn key_range(&self) -> std::ops::Range<usize> {
        self.key_off as usize..(self.key_off as usize + self.key_len as usize)
    }
}

trait Bucket: Copy + Default {
    const WAYS: usize;
    fn slots(&self) -> &[CacheSlot];
    fn slots_mut(&mut self) -> &mut [CacheSlot];
}

macro_rules! bucket {
    ($name:ident, $ways:expr $(, $align:meta)?) => {
        #[derive(Clone, Copy, Default)]
        $(#[$align])?
        struct $name([CacheSlot; $ways]);

        impl Bucket for $name {
            const WAYS: usize = $ways;
            fn slots(&self) -> &[CacheSlot] {
                &self.0
            }
            fn slots_mut(&mut self) -> &mut [CacheSlot] {
                &mut self.0
            }
        }
    };
}

bucket!(Bucket1, 1);
bucket!(Bucket4, 4, repr(align(64)));
bucket!(Bucket8, 8, repr(align(64)));

struct FixedCache<B> {
    hasher: RandomState,
    buckets: Box<[B]>,
    key_bytes: Vec<u8>,
    ids: Vec<u32>,
    bucket_mask: u64,
    max_length: usize,
}

impl<B: Bucket> FixedCache<B> {
    fn new(capacity: usize, max_length: usize) -> Self {
        let n_buckets = (capacity.next_power_of_two() / B::WAYS).max(1);
        Self {
            hasher: fixed_hasher(),
            buckets: vec![B::default(); n_buckets].into_boxed_slice(),
            ids: Vec::with_capacity(256),
            key_bytes: Vec::with_capacity(1024),
            bucket_mask: (n_buckets as u64) - 1,
            max_length,
        }
    }

    fn locate(&self, key: &[u8]) -> (usize, u32) {
        let hash = self.hasher.hash_one(key);
        ((hash & self.bucket_mask) as usize, (hash >> 32) as u32 | 1)
    }
}

impl<B: Bucket> CacheVariant for FixedCache<B> {
    fn get(&mut self, word: &str) -> Option<&[u32]> {
        let key = word.as_bytes();
        if key.len() > self.max_length {
            return None;
        }
        let (bucket_idx, tag) = self.locate(key);
        for slot in self.buckets[bucket_idx].slots() {
            if slot.tag == tag && key == &self.key_bytes[slot.key_range()] {
                return Some(&self.ids[slot.id_range()]);
            }
        }
        None
    }

    fn insert(&mut self, word: &str, ids: &[u32]) -> InsertOutcome {
        let key = word.as_bytes();
        if key.len() > self.max_length {
            return InsertOutcome::RejectedLen;
        }
        let (bucket_idx, tag) = self.locate(key);
        let Some(slot) = self.buckets[bucket_idx]
            .slots()
            .iter()
            .position(|slot| slot.tag == 0)
        else {
            return InsertOutcome::RejectedFull;
        };
        self.buckets[bucket_idx].slots_mut()[slot] = CacheSlot {
            tag,
            key_off: self.key_bytes.len() as u32,
            key_len: key.len() as u16,
            ids_off: self.ids.len() as u32,
            ids_len: ids.len() as u16,
        };
        self.key_bytes.extend_from_slice(key);
        self.ids.extend_from_slice(ids);
        InsertOutcome::Stored
    }

    fn bucket_load(&self, word: &str) -> Option<usize> {
        let (bucket_idx, _) = self.locate(word.as_bytes());
        Some(
            self.buckets[bucket_idx]
                .slots()
                .iter()
                .filter(|s| s.tag != 0)
                .count(),
        )
    }

    fn probe_tags(&self, word: &str) -> Option<u32> {
        let key = word.as_bytes();
        if key.len() > self.max_length {
            return Some(0);
        }
        let (bucket_idx, tag) = self.locate(key);
        Some(
            self.buckets[bucket_idx]
                .slots()
                .iter()
                .filter(|s| s.tag == tag)
                .count() as u32,
        )
    }

    fn occupied(&self) -> usize {
        self.buckets
            .iter()
            .flat_map(Bucket::slots)
            .filter(|s| s.tag != 0)
            .count()
    }
    fn slot_count(&self) -> usize {
        self.buckets.len() * B::WAYS
    }
    fn memory(&self) -> MemBreakdown {
        MemBreakdown {
            table: self.buckets.len() * size_of::<B>(),
            keys: self.key_bytes.capacity(),
            ids: self.ids.capacity() * size_of::<u32>(),
        }
    }
}

/// The poach experiment: WordCache's 4-way bucket crossed with FlatCache's
/// retention idea, at zero slot growth. The reuse flag lives in a spare top
/// bit of `key_len` (sound while max_length < 2^15); a hit sets its slot's
/// flag. A full-bucket insert scans CLOCK-style: set flags are cleared in
/// passing and the first unflagged slot is evicted; if every slot was flagged
/// the insert is rejected and the scan itself has aged the bucket — persistent
/// hotness must re-earn its place, and a genuinely hot bucket still freezes.
/// Evicted entries orphan their arena bytes; when an arena outgrows its budget
/// the live entries are compacted into fresh buffers.
const REUSE: u16 = 1 << 15;

struct BucketCull {
    hasher: RandomState,
    buckets: Box<[Bucket4]>,
    key_bytes: Vec<u8>,
    ids: Vec<u32>,
    bucket_mask: u64,
    max_length: usize,
    key_budget: usize,
    ids_budget: usize,
    culls: usize,
}

impl BucketCull {
    fn new(capacity: usize, max_length: usize) -> Self {
        let slots = capacity.next_power_of_two();
        let n_buckets = (slots / 4).max(1);
        Self {
            hasher: fixed_hasher(),
            buckets: vec![Bucket4::default(); n_buckets].into_boxed_slice(),
            key_bytes: Vec::with_capacity(1024),
            ids: Vec::with_capacity(256),
            bucket_mask: (n_buckets as u64) - 1,
            max_length,
            key_budget: slots * 16,
            ids_budget: slots * 8,
            culls: 0,
        }
    }

    fn locate(&self, key: &[u8]) -> (usize, u32) {
        let hash = self.hasher.hash_one(key);
        ((hash & self.bucket_mask) as usize, (hash >> 32) as u32 | 1)
    }

    fn compact(&mut self) {
        self.culls += 1;
        let mut keys = Vec::with_capacity(self.key_budget);
        let mut ids = Vec::with_capacity(self.ids_budget);
        for bucket in self.buckets.iter_mut() {
            for s in bucket.slots_mut() {
                if s.tag == 0 {
                    continue;
                }
                let klen = (s.key_len & !REUSE) as usize;
                let koff = keys.len() as u32;
                let ioff = ids.len() as u32;
                keys.extend_from_slice(&self.key_bytes[s.key_off as usize..s.key_off as usize + klen]);
                ids.extend_from_slice(&self.ids[s.ids_off as usize..s.ids_off as usize + s.ids_len as usize]);
                s.key_off = koff;
                s.ids_off = ioff;
            }
        }
        self.key_bytes = keys;
        self.ids = ids;
    }
}

impl CacheVariant for BucketCull {
    fn get(&mut self, word: &str) -> Option<&[u32]> {
        let key = word.as_bytes();
        if key.len() > self.max_length {
            return None;
        }
        let (bucket_idx, tag) = self.locate(key);
        let mut found = None;
        for (i, s) in self.buckets[bucket_idx].slots().iter().enumerate() {
            let klen = (s.key_len & !REUSE) as usize;
            if s.tag == tag && key == &self.key_bytes[s.key_off as usize..s.key_off as usize + klen]
            {
                found = Some((i, s.ids_off as usize, s.ids_len as usize));
                break;
            }
        }
        let (i, off, len) = found?;
        let slot = &mut self.buckets[bucket_idx].slots_mut()[i];
        if slot.key_len & REUSE == 0 {
            slot.key_len |= REUSE;
        }
        Some(&self.ids[off..off + len])
    }

    fn insert(&mut self, word: &str, ids: &[u32]) -> InsertOutcome {
        let key = word.as_bytes();
        if key.len() > self.max_length {
            return InsertOutcome::RejectedLen;
        }
        if self.key_bytes.len() + key.len() > self.key_budget
            || self.ids.len() + ids.len() > self.ids_budget
        {
            self.compact();
        }
        let (bucket_idx, tag) = self.locate(key);
        let slots = self.buckets[bucket_idx].slots();
        let (victim, outcome) = match slots.iter().position(|s| s.tag == 0) {
            Some(empty) => (Some(empty), InsertOutcome::Stored),
            None => {
                // Evict only a slot that was never reused; flags survive the
                // scan. Rejection is the aging tick: when every slot has
                // earned its place the bucket freezes for this round, and all
                // flags reset so each entry must re-earn it before the next
                // conflict. Clearing flags during the scan instead (classic
                // CLOCK) measured worse: Zipf-tail one-shots kept evicting
                // warm words.
                match self.buckets[bucket_idx].slots().iter().position(|s| s.key_len & REUSE == 0)
                {
                    Some(i) => (Some(i), InsertOutcome::Evicted),
                    None => {
                        for s in self.buckets[bucket_idx].slots_mut() {
                            s.key_len &= !REUSE;
                        }
                        (None, InsertOutcome::RejectedFull)
                    }
                }
            }
        };
        let Some(slot) = victim else {
            return outcome;
        };
        self.buckets[bucket_idx].slots_mut()[slot] = CacheSlot {
            tag,
            key_off: self.key_bytes.len() as u32,
            key_len: key.len() as u16,
            ids_off: self.ids.len() as u32,
            ids_len: ids.len() as u16,
        };
        self.key_bytes.extend_from_slice(key);
        self.ids.extend_from_slice(ids);
        outcome
    }

    fn culls(&self) -> Option<usize> {
        Some(self.culls)
    }
    fn bucket_load(&self, word: &str) -> Option<usize> {
        let (bucket_idx, _) = self.locate(word.as_bytes());
        Some(self.buckets[bucket_idx].slots().iter().filter(|s| s.tag != 0).count())
    }
    fn probe_tags(&self, word: &str) -> Option<u32> {
        let key = word.as_bytes();
        if key.len() > self.max_length {
            return Some(0);
        }
        let (bucket_idx, tag) = self.locate(key);
        Some(self.buckets[bucket_idx].slots().iter().filter(|s| s.tag == tag).count() as u32)
    }
    fn occupied(&self) -> usize {
        self.buckets.iter().flat_map(Bucket::slots).filter(|s| s.tag != 0).count()
    }
    fn slot_count(&self) -> usize {
        self.buckets.len() * 4
    }
    fn memory(&self) -> MemBreakdown {
        MemBreakdown {
            table: self.buckets.len() * size_of::<Bucket4>(),
            keys: self.key_bytes.capacity(),
            ids: self.ids.capacity() * size_of::<u32>(),
        }
    }
}

// Replica of PR #2234's FlatCache (branch perf/fused-split-cache,
// tk-encode/src/models/bpe/flat_cache.rs). Keep in sync by hand. Open-addressed
// linear probing, <= 75% load; keys <= 15 bytes are packed into a u128 (length
// in the top byte) and confirmed with one register compare under a CRC hash;
// longer keys hash with ahash and byte-verify against the arena. When the table
// (or an arena) fills, `cull` compacts into the spare buffers keeping only
// entries reused since the last cull, halving their frequency; if nearly
// everything survived, it clears instead. No key-length cap by design.

#[derive(Clone, Copy)]
struct FlatSlot {
    hash: u64,
    key: u128,
    koff: u32,
    ioff: u32,
    klen: u16,
    ilen: u16,
    freq: u8,
}
const FLAT_EMPTY: FlatSlot =
    FlatSlot { hash: 0, key: 0, koff: 0, ioff: 0, klen: 0, ilen: 0, freq: 0 };

fn pack_key(p: &[u8]) -> u128 {
    if p.is_empty() || p.len() > 15 {
        return 0;
    }
    let mut b = [0u8; 16];
    b[..p.len()].copy_from_slice(p);
    u128::from_le_bytes(b) | ((p.len() as u128) << 120)
}

fn crc_key_hash(key: u128) -> u64 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::__crc32cd;
        __crc32cd(__crc32cd(0, key as u64), (key >> 64) as u64) as u64
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut h = (key as u64).wrapping_mul(0xff51afd7ed558ccd)
            ^ ((key >> 64) as u64).wrapping_mul(0xc4ceb9fe1a85ec53);
        h ^= h >> 33;
        h
    }
}

struct FlatCacheReplica {
    slots: Box<[FlatSlot]>,
    slots2: Box<[FlatSlot]>,
    kbytes: Vec<u8>,
    kbytes2: Vec<u8>,
    ids: Vec<u32>,
    ids2: Vec<u32>,
    mask: usize,
    count: usize,
    culls: usize,
    hasher: RandomState,
}

impl FlatCacheReplica {
    fn new(capacity: usize) -> Self {
        let slots_n = capacity.next_power_of_two();
        Self {
            slots: vec![FLAT_EMPTY; slots_n].into_boxed_slice(),
            slots2: vec![FLAT_EMPTY; slots_n].into_boxed_slice(),
            kbytes: Vec::with_capacity(slots_n * 16),
            kbytes2: Vec::with_capacity(slots_n * 16),
            ids: Vec::with_capacity(slots_n * 8),
            ids2: Vec::with_capacity(slots_n * 8),
            mask: slots_n - 1,
            count: 0,
            culls: 0,
            hasher: RandomState::with_seeds(0xC0FFEE, 0xBADF00D, 0xDEADBEEF, 0x1337),
        }
    }

    fn hash_of(&self, p: &[u8], key: u128) -> u64 {
        if key != 0 { crc_key_hash(key) } else { self.hasher.hash_one(p) }
    }

    fn clear(&mut self) {
        for s in self.slots.iter_mut() {
            s.klen = 0;
        }
        self.kbytes.clear();
        self.ids.clear();
        self.count = 0;
    }

    fn cull(&mut self) {
        self.culls += 1;
        let Self { slots, slots2, kbytes, kbytes2, ids, ids2, mask, .. } = self;
        for s in slots2.iter_mut() {
            s.klen = 0;
        }
        kbytes2.clear();
        ids2.clear();
        let mut kept = 0usize;
        for s in slots.iter() {
            if s.klen == 0 || s.freq == 0 {
                continue;
            }
            let koff = kbytes2.len() as u32;
            let ioff = ids2.len() as u32;
            kbytes2.extend_from_slice(&kbytes[s.koff as usize..s.koff as usize + s.klen as usize]);
            ids2.extend_from_slice(&ids[s.ioff as usize..s.ioff as usize + s.ilen as usize]);
            let mut i = (s.hash as usize) & *mask;
            while slots2[i].klen != 0 {
                i = (i + 1) & *mask;
            }
            slots2[i] = FlatSlot { koff, ioff, freq: s.freq >> 1, ..*s };
            kept += 1;
        }
        std::mem::swap(slots, slots2);
        std::mem::swap(kbytes, kbytes2);
        std::mem::swap(ids, ids2);
        self.count = kept;
        if kept * 4 >= self.slots.len() * 3 {
            self.clear();
        }
    }
}

impl CacheVariant for FlatCacheReplica {
    fn get(&mut self, word: &str) -> Option<&[u32]> {
        let p = word.as_bytes();
        let key = pack_key(p);
        let h = self.hash_of(p, key);
        let mut i = (h as usize) & self.mask;
        loop {
            let s = self.slots[i];
            if s.klen == 0 {
                return None;
            }
            let confirmed = if key != 0 {
                s.key == key
            } else {
                s.hash == h
                    && s.klen as usize == p.len()
                    && self.kbytes[s.koff as usize..s.koff as usize + s.klen as usize] == *p
            };
            if confirmed {
                self.slots[i].freq = self.slots[i].freq.saturating_add(1);
                return Some(&self.ids[s.ioff as usize..s.ioff as usize + s.ilen as usize]);
            }
            i = (i + 1) & self.mask;
        }
    }

    fn insert(&mut self, word: &str, ids: &[u32]) -> InsertOutcome {
        let p = word.as_bytes();
        if ids.is_empty() || p.len() > u16::MAX as usize || ids.len() > u16::MAX as usize {
            return InsertOutcome::RejectedLen;
        }
        if self.kbytes.len() + p.len() > self.kbytes.capacity()
            || self.ids.len() + ids.len() > self.ids.capacity()
            || self.count * 4 >= self.slots.len() * 3
        {
            self.cull();
        }
        let key = pack_key(p);
        let h = self.hash_of(p, key);
        let (koff, ioff) = (self.kbytes.len() as u32, self.ids.len() as u32);
        self.kbytes.extend_from_slice(p);
        self.ids.extend_from_slice(ids);
        let mut i = (h as usize) & self.mask;
        while self.slots[i].klen != 0 {
            i = (i + 1) & self.mask;
        }
        self.slots[i] = FlatSlot {
            hash: h,
            key,
            koff,
            ioff,
            klen: p.len() as u16,
            ilen: ids.len() as u16,
            freq: 0,
        };
        self.count += 1;
        InsertOutcome::Stored
    }

    fn culls(&self) -> Option<usize> {
        Some(self.culls)
    }
    // Displacement in open addressing isn't comparable to bucket collisions;
    // report null rather than a misleading number.
    fn bucket_load(&self, _: &str) -> Option<usize> {
        None
    }
    fn probe_tags(&self, _: &str) -> Option<u32> {
        None
    }
    fn occupied(&self) -> usize {
        self.count
    }
    fn slot_count(&self) -> usize {
        self.slots.len()
    }
    /// Both generations are allocated, so both are counted.
    fn memory(&self) -> MemBreakdown {
        MemBreakdown {
            table: (self.slots.len() + self.slots2.len()) * size_of::<FlatSlot>(),
            keys: self.kbytes.capacity() + self.kbytes2.capacity(),
            ids: (self.ids.capacity() + self.ids2.capacity()) * size_of::<u32>(),
        }
    }
}

// ── workload ────────────────────────────────────────────────────────────────

/// The word stream a fixture produces for a given tokenizer: per chunk, the
/// normalized text plus the pre-token spans into it — exactly what
/// `encode_generic` feeds the model, minus added-token segmentation (chunks
/// where that matters are filtered out in `bench_fixture`).
struct Stream {
    chunks: Vec<(String, Vec<Span>)>,
    words: usize,
    bytes: usize,
}

fn build_stream(tok: &Tokenizer, chunks: &[String]) -> tk_encode::Result<Stream> {
    let pre_tokenizer: PipelinePreTokenizer = tok
        .get_pre_tokenizer()
        .cloned()
        .map(TryInto::try_into)
        .transpose()?
        .unwrap_or(PipelinePreTokenizer::None);
    let mut out = Vec::with_capacity(chunks.len());
    let mut words = 0;
    for chunk in chunks {
        let normalized: Cow<str> = match tok.get_normalizer() {
            Some(n) => n.normalize(chunk)?,
            None => Cow::Borrowed(chunk),
        };
        let mut spans = Vec::new();
        pre_tokenizer.pre_tokenize(&normalized, &mut spans)?;
        words += spans.len();
        out.push((normalized.into_owned(), spans));
    }
    Ok(Stream {
        bytes: chunks.iter().map(String::len).sum(),
        chunks: out,
        words,
    })
}

fn make_chunks(text: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut cur = String::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        if !cur.is_empty() {
            cur.push('\n');
        }
        cur.push_str(line);
        if cur.len() >= CHUNK_BYTES {
            chunks.push(std::mem::take(&mut cur));
            if chunks.len() == MAX_CHUNKS {
                return chunks;
            }
        }
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

// ── measurement ─────────────────────────────────────────────────────────────

#[derive(Default)]
struct PassStats {
    secs: f64,
    hits: usize,
    misses: usize,
}

/// One untimed-per-op pass of the word stream through `cache`, tokenizing
/// misses with the real (cache-free) model. Mirrors the shipped hit path in
/// `PipelineBPE::tokenize_pipeline`. With `gate`, also accumulates the full id
/// sequence for correctness checking. Used for throughput; per-op latency
/// comes from `instrumented_pass`.
fn replay_pass(
    cache: &mut dyn CacheVariant,
    stream: &Stream,
    model: &PipelineModel,
    scratch: &mut <PipelineModel as Model>::Scratch,
    mut gate: Option<&mut Vec<u32>>,
) -> tk_encode::Result<PassStats> {
    let mut stats = PassStats::default();
    let mut buf: Vec<u32> = Vec::new();
    let mut tokens: Vec<PipelineToken> = Vec::new();
    let mut miss_ids: Vec<u32> = Vec::new();
    let start = Instant::now();
    for (text, spans) in &stream.chunks {
        buf.clear();
        for span in spans {
            let word = &text[span.start as usize..span.end as usize];
            if let Some(ids) = cache.get(word) {
                stats.hits += 1;
                buf.extend_from_slice(ids);
            } else {
                stats.misses += 1;
                tokens.clear();
                model.tokenize_pipeline(word, scratch, &mut tokens)?;
                miss_ids.clear();
                miss_ids.extend(tokens.iter().map(|t| t.id));
                cache.insert(word, &miss_ids);
                buf.extend_from_slice(&miss_ids);
            }
        }
        black_box(&buf);
        if let Some(gate) = gate.as_deref_mut() {
            gate.extend_from_slice(&buf);
        }
    }
    stats.secs = start.elapsed().as_secs_f64();
    Ok(stats)
}

#[derive(Default)]
struct OpSamples {
    hit: Vec<u64>,
    miss: Vec<u64>,
    insert: Vec<u64>,
}

#[derive(Default)]
struct InsertCounters {
    attempts: usize,
    stored: usize,
    evicted: usize,
    rejected_full: usize,
    rejected_len: usize,
    collisions: usize,
    collisions_observable: bool,
}

/// Same replay loop with every cache op individually timed, plus insert
/// outcome and bucket-collision accounting. The timing wraps only the cache
/// call — never the model. Throughput must not be read off this pass.
fn instrumented_pass(
    cache: &mut dyn CacheVariant,
    stream: &Stream,
    model: &PipelineModel,
    scratch: &mut <PipelineModel as Model>::Scratch,
    samples: &mut OpSamples,
    inserts: &mut InsertCounters,
    mut gate: Option<&mut Vec<u32>>,
) -> tk_encode::Result<()> {
    let mut buf: Vec<u32> = Vec::new();
    let mut tokens: Vec<PipelineToken> = Vec::new();
    let mut miss_ids: Vec<u32> = Vec::new();
    for (text, spans) in &stream.chunks {
        buf.clear();
        for span in spans {
            let word = &text[span.start as usize..span.end as usize];
            let t0 = Instant::now();
            let hit = cache.get(word);
            let dt = t0.elapsed().as_nanos() as u64;
            if let Some(ids) = hit {
                samples.hit.push(dt);
                buf.extend_from_slice(ids);
            } else {
                samples.miss.push(dt);
                tokens.clear();
                model.tokenize_pipeline(word, scratch, &mut tokens)?;
                miss_ids.clear();
                miss_ids.extend(tokens.iter().map(|t| t.id));
                inserts.attempts += 1;
                let t0 = Instant::now();
                let outcome = cache.insert(word, &miss_ids);
                let dt = t0.elapsed().as_nanos() as u64;
                match outcome {
                    InsertOutcome::Stored | InsertOutcome::Evicted => {
                        samples.insert.push(dt);
                        if outcome == InsertOutcome::Evicted {
                            inserts.evicted += 1;
                        } else {
                            inserts.stored += 1;
                        }
                        if let Some(load) = cache.bucket_load(word) {
                            inserts.collisions_observable = true;
                            if load > 1 {
                                inserts.collisions += 1;
                            }
                        }
                    }
                    InsertOutcome::RejectedFull => inserts.rejected_full += 1,
                    InsertOutcome::RejectedLen => inserts.rejected_len += 1,
                }
                buf.extend_from_slice(&miss_ids);
            }
        }
        black_box(&buf);
        if let Some(gate) = gate.as_deref_mut() {
            gate.extend_from_slice(&buf);
        }
    }
    Ok(())
}

/// Count tag matches vs real hits over one pass: the excess is 32-bit tag
/// false positives (each one costs a wasted key compare on the shipped probe).
fn tag_pass(cache: &mut dyn CacheVariant, stream: &Stream) -> Option<(u64, u64, u64)> {
    let mut gets = 0u64;
    let mut hits = 0u64;
    let mut tags = 0u64;
    for (text, spans) in &stream.chunks {
        for span in spans {
            let word = &text[span.start as usize..span.end as usize];
            tags += cache.probe_tags(word)? as u64;
            gets += 1;
            hits += cache.get(word).is_some() as u64;
        }
    }
    Some((gets, tags - hits, hits))
}

fn percentiles(mut xs: Vec<u64>) -> Value {
    if xs.is_empty() {
        return Value::Null;
    }
    xs.sort_unstable();
    let pick = |p: f64| xs[((xs.len() - 1) as f64 * p) as usize];
    json!({
        "n": xs.len(),
        "mean": xs.iter().sum::<u64>() as f64 / xs.len() as f64,
        "p50": pick(0.50),
        "p90": pick(0.90),
        "p99": pick(0.99),
        "max": xs[xs.len() - 1],
    })
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

/// Measured cost and granularity of the per-op timer, to read the latency
/// percentiles against: an op that reads as 0 ns is simply below resolution.
fn timer_calibration() -> Value {
    let mut deltas: Vec<u64> = (0..100_000)
        .map(|_| {
            let t0 = Instant::now();
            black_box(t0.elapsed().as_nanos() as u64)
        })
        .collect();
    deltas.sort_unstable();
    let resolution = deltas.iter().copied().find(|&d| d > 0).unwrap_or(0);
    json!({
        "overhead_ns_p50": deltas[deltas.len() / 2],
        "resolution_ns": resolution,
    })
}

/// Encode all chunks through a fresh pipeline whose legacy BPE cache is
/// resized to `capacity` (0 disables it — `TryFrom` then builds no word
/// cache). Returns median MB/s over `reps` warm passes and the id sequence.
fn end_to_end(
    tok: &Tokenizer,
    capacity: usize,
    chunks: &[String],
    bytes: usize,
    reps: usize,
) -> tk_encode::Result<(f64, Vec<u32>)> {
    let mut tok = tok.clone();
    if let ModelWrapper::BPE(bpe) = tok.get_model_mut() {
        bpe.resize_cache(capacity);
    }
    let pipeline = PipelineTokenizer::try_from(&tok)?;
    let mut ids = Vec::new();
    for chunk in chunks {
        ids.extend(pipeline.encode(chunk, false)?.iter().map(|t| t.id));
    }
    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        for chunk in chunks {
            black_box(pipeline.encode(chunk, false)?);
        }
        times.push(start.elapsed().as_secs_f64());
    }
    Ok((bytes as f64 / median(times) / 1e6, ids))
}

fn build_variant(name: &str, capacity: usize, max_length: usize) -> Box<dyn CacheVariant> {
    match name {
        "naive_hashmap" => Box::new(NaiveCache::new(capacity, max_length)),
        "direct_mapped" => Box::new(FixedCache::<Bucket1>::new(capacity, max_length)),
        "assoc_4way" => Box::new(FixedCache::<Bucket4>::new(capacity, max_length)),
        "assoc_8way" => Box::new(FixedCache::<Bucket8>::new(capacity, max_length)),
        "flat_cache" => Box::new(FlatCacheReplica::new(capacity)),
        "bucket_cull" => Box::new(BucketCull::new(capacity, max_length)),
        other => unreachable!("{other}"),
    }
}

const VARIANTS: &[&str] = &[
    "naive_hashmap",
    "direct_mapped",
    "assoc_4way",
    "assoc_8way",
    "flat_cache",
    "bucket_cull",
];

#[allow(clippy::too_many_arguments)]
fn bench_fixture(
    tok: &Tokenizer,
    name: &str,
    group: &str,
    chunks: &[String],
    capacities: &[usize],
    max_lengths: &[usize],
    reps: usize,
) -> tk_encode::Result<Value> {
    // The replay model comes from a cache-disabled pipeline so misses run the
    // bare merge loop and the only cache in play is the variant under test.
    let mut off_tok = tok.clone();
    if let ModelWrapper::BPE(bpe) = off_tok.get_model_mut() {
        bpe.resize_cache(0);
    }
    let pipeline_off = PipelineTokenizer::try_from(&off_tok)?;
    let model = pipeline_off.get_model();
    let mut scratch = model.init_scratch();

    // Replay has no added-token segmentation, so a chunk that happens to
    // contain one of the model's added tokens (e.g. `<think>` in the agentic
    // traces under deepseek) would encode differently through it. Drop those
    // chunks and report the count — the cache comparison doesn't need them.
    let full = build_stream(tok, chunks)?;
    let mut kept_chunks = Vec::new();
    let mut kept_stream = Vec::new();
    let mut reference = Vec::new();
    for (chunk, entry) in chunks.iter().zip(full.chunks) {
        let single = Stream {
            bytes: chunk.len(),
            words: entry.1.len(),
            chunks: vec![entry],
        };
        let mut replay_ids = Vec::new();
        replay_pass(&mut NoCache, &single, model, &mut scratch, Some(&mut replay_ids))?;
        let pipeline_ids: Vec<u32> = pipeline_off
            .encode(chunk, false)?
            .iter()
            .map(|t| t.id)
            .collect();
        if replay_ids == pipeline_ids {
            kept_chunks.push(chunk.clone());
            kept_stream.extend(single.chunks);
            reference.extend(pipeline_ids);
        }
    }
    let chunks_skipped = chunks.len() - kept_chunks.len();
    if kept_chunks.is_empty() {
        return Ok(json!({
            "fixture": name,
            "group": group,
            "reason": "all chunks diverge from the pipeline (added tokens?)",
        }));
    }
    let stream = Stream {
        bytes: kept_chunks.iter().map(String::len).sum(),
        words: kept_stream.iter().map(|(_, s)| s.len()).sum(),
        chunks: kept_stream,
    };

    let mut unique: AHashSet<&str> = AHashSet::new();
    for (text, spans) in &stream.chunks {
        unique.extend(
            spans
                .iter()
                .map(|s| &text[s.start as usize..s.end as usize]),
        );
    }
    let unique_cacheable: Value = max_lengths
        .iter()
        .map(|ml| {
            (
                ml.to_string(),
                unique.iter().filter(|w| w.len() <= *ml).count().into(),
            )
        })
        .collect::<serde_json::Map<String, Value>>()
        .into();

    // No-cache floor, measured once: capacity/max_length can't affect it.
    let baseline_mb_s = {
        let mut times = Vec::with_capacity(reps);
        for _ in 0..reps {
            times.push(replay_pass(&mut NoCache, &stream, model, &mut scratch, None)?.secs);
        }
        stream.bytes as f64 / median(times) / 1e6
    };

    let (off_mb_s, _) = end_to_end(tok, 0, &kept_chunks, stream.bytes, reps)?;
    let mut e2e_on = Vec::new();
    let mut e2e_ids_match = true;
    for &cap in capacities {
        let (mb_s, ids) = end_to_end(tok, cap, &kept_chunks, stream.bytes, reps)?;
        e2e_ids_match &= ids == reference;
        e2e_on.push(json!({
            "capacity": cap,
            "mb_per_s": mb_s,
            "speedup": mb_s / off_mb_s,
        }));
    }

    let mut results = Vec::new();
    for &capacity in capacities {
        for (mi, &max_length) in max_lengths.iter().enumerate() {
            for vname in VARIANTS {
                // flat_cache has no key-length cap: one run per capacity.
                if *vname == "flat_cache" && mi != 0 {
                    continue;
                }
                let mut cache = build_variant(vname, capacity, max_length);

                // Cold: instrumented, and the correctness gate.
                let mut cold_samples = OpSamples::default();
                let mut inserts = InsertCounters::default();
                let mut replay_ids = Vec::with_capacity(reference.len());
                instrumented_pass(
                    cache.as_mut(),
                    &stream,
                    model,
                    &mut scratch,
                    &mut cold_samples,
                    &mut inserts,
                    Some(&mut replay_ids),
                )?;
                let ids_match = replay_ids == reference;

                // Warm throughput, untimed per op.
                let mut times = Vec::with_capacity(reps);
                let mut warm = PassStats::default();
                for _ in 0..reps {
                    warm = replay_pass(cache.as_mut(), &stream, model, &mut scratch, None)?;
                    times.push(warm.secs);
                }
                let mb_s = stream.bytes as f64 / median(times) / 1e6;

                // Steady state: instrumented pass for hit/miss latency (inserts
                // still occur — steady misses retry — and land in the same
                // counters), then the tag-quality pass.
                let mut steady_samples = OpSamples::default();
                instrumented_pass(
                    cache.as_mut(),
                    &stream,
                    model,
                    &mut scratch,
                    &mut steady_samples,
                    &mut inserts,
                    None,
                )?;
                let tag_stats = tag_pass(cache.as_mut(), &stream);

                let mem = cache.memory();
                let occupied = cache.occupied();
                let slots = cache.slot_count();
                let rejected = inserts.rejected_full + inserts.rejected_len;
                results.push(json!({
                    "variant": vname,
                    "capacity": capacity,
                    "max_length": if *vname == "flat_cache" { Value::Null } else { max_length.into() },
                    "culls": cache.culls(),
                    "ids_match": ids_match,
                    "warm_mb_per_s": mb_s,
                    "speedup_vs_none": mb_s / baseline_mb_s,
                    "steady": {
                        "hit_rate": warm.hits as f64 / stream.words as f64,
                        "hits": warm.hits,
                        "misses": warm.misses,
                    },
                    "inserts": {
                        "attempts": inserts.attempts,
                        "stored": inserts.stored,
                        "evicted": inserts.evicted,
                        "rejected_full": inserts.rejected_full,
                        "rejected_len": inserts.rejected_len,
                        "failure_rate": rejected as f64 / inserts.attempts.max(1) as f64,
                        "collisions": inserts.collisions_observable.then_some(inserts.collisions),
                        "collision_rate": inserts.collisions_observable.then(|| {
                            inserts.collisions as f64
                                / (inserts.stored + inserts.evicted).max(1) as f64
                        }),
                    },
                    "occupancy": {
                        "occupied": occupied,
                        "slots": slots,
                        "rate": occupied as f64 / slots.max(1) as f64,
                    },
                    "memory_bytes": {
                        "table": mem.table,
                        "keys": mem.keys,
                        "ids": mem.ids,
                        "total": mem.table + mem.keys + mem.ids,
                    },
                    "latency_ns": {
                        "hit_steady": percentiles(steady_samples.hit),
                        "miss_steady": percentiles(steady_samples.miss),
                        "miss_cold": percentiles(cold_samples.miss),
                        "insert": percentiles({
                            let mut all = cold_samples.insert;
                            all.extend(steady_samples.insert);
                            all
                        }),
                    },
                    "tag_quality": tag_stats.map(|(gets, false_tags, _)| json!({
                        "gets": gets,
                        "false_tag_matches": false_tags,
                    })),
                }));
            }
        }
    }

    Ok(json!({
        "fixture": name,
        "group": group,
        "bytes": stream.bytes,
        "chunks": kept_chunks.len(),
        "chunks_skipped": chunks_skipped,
        "words": stream.words,
        "unique_words": unique.len(),
        "unique_cacheable": unique_cacheable,
        "replay_baseline_mb_per_s": baseline_mb_s,
        "end_to_end": {
            "off_mb_per_s": off_mb_s,
            "on": e2e_on,
            "ids_match": e2e_ids_match,
        },
        "variants": results,
    }))
}

// ── driver ──────────────────────────────────────────────────────────────────

struct Args {
    models: Option<Vec<String>>,
    fixtures: Option<Vec<String>>,
    capacities: Vec<usize>,
    max_lengths: Vec<usize>,
    reps: usize,
    out: Option<String>,
}

fn parse_args() -> Args {
    let mut args = Args {
        models: None,
        fixtures: None,
        capacities: DEFAULT_CAPACITIES.to_vec(),
        max_lengths: DEFAULT_MAX_LENGTHS.to_vec(),
        reps: 3,
        out: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let value = it.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        let strs = || value.split(',').map(str::to_lowercase).collect();
        let nums = || value.split(',').map(|s| s.parse().unwrap()).collect();
        match flag.as_str() {
            "--models" => args.models = Some(strs()),
            "--fixtures" => args.fixtures = Some(strs()),
            "--capacities" => args.capacities = nums(),
            "--max-lengths" => args.max_lengths = nums(),
            "--reps" => args.reps = value.parse().unwrap(),
            "--out" => args.out = Some(value),
            other => panic!("unknown flag {other}"),
        }
    }
    args
}

fn keep(filter: &Option<Vec<String>>, name: &str) -> bool {
    filter
        .as_ref()
        .is_none_or(|f| f.iter().any(|s| name.to_lowercase().contains(s)))
}

fn main() {
    let args = parse_args();
    if cfg!(debug_assertions) {
        eprintln!("WARNING: debug build — numbers are meaningless, use --release");
    }

    let manifest: Vec<Value> =
        serde_json::from_str(&std::fs::read_to_string(MANIFEST).unwrap()).unwrap();

    let mut fixtures: Vec<(String, &str, Vec<String>)> = Vec::new();
    for group in ["lang", "modalities"] {
        let dir = Path::new(DATA_DIR).join("fixtures").join(group);
        let mut paths: Vec<_> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("{}: {e} — run `make fixtures` first", dir.display()))
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|x| x == "txt"))
            .collect();
        paths.sort();
        for path in paths {
            let name = path.file_stem().unwrap().to_str().unwrap().to_string();
            // The added_* fixtures only exercise added-token segmentation,
            // which replay skips; without injected tokens they are plain text.
            if name.starts_with("added_") || !keep(&args.fixtures, &name) {
                continue;
            }
            fixtures.push((name, group, make_chunks(&std::fs::read_to_string(path).unwrap())));
        }
    }

    let mut models = Vec::new();
    for entry in &manifest {
        let name = entry["name"].as_str().unwrap();
        if !keep(&args.models, name) {
            continue;
        }
        let path = Path::new(DATA_DIR).join(entry["file"].as_str().unwrap());
        eprintln!("== {name} ==");

        let skip = |reason: String| {
            eprintln!("  skipped: {reason}");
            json!({ "model": name, "reason": reason, "fixtures": [] })
        };
        let tok = match Tokenizer::from_file(&path) {
            Ok(t) => t,
            Err(e) => {
                models.push(skip(format!("load error: {e}")));
                continue;
            }
        };
        if !matches!(tok.get_model(), ModelWrapper::BPE(_)) {
            models.push(skip("non-BPE model — the word cache is BPE-only".into()));
            continue;
        }
        match PipelineTokenizer::try_from(&tok).and_then(|p| p.encode(PROBE, false)) {
            Ok(_) => {}
            Err(e) => {
                models.push(skip(format!("pipeline unsupported: {e}")));
                continue;
            }
        }

        let mut results = Vec::new();
        for (fname, group, chunks) in &fixtures {
            eprint!("  {fname} ... ");
            match bench_fixture(
                &tok,
                fname,
                group,
                chunks,
                &args.capacities,
                &args.max_lengths,
                args.reps,
            ) {
                Ok(v) => {
                    eprintln!("ok");
                    results.push(v);
                }
                Err(e) => {
                    eprintln!("failed: {e}");
                    results.push(json!({ "fixture": fname, "reason": e.to_string() }));
                }
            }
        }
        models.push(json!({
            "model": name,
            "desc": entry.get("desc").and_then(Value::as_str).unwrap_or(""),
            "fixtures": results,
        }));
    }

    let git = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    let report = json!({
        "meta": {
            "generated_unix": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            "commit": git,
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "release": !cfg!(debug_assertions),
            "single_threaded": true,
            "capacities": args.capacities,
            "max_lengths": args.max_lengths,
            "shipped": { "capacity": 65_536, "max_length": 256 },
            "chunk_bytes": CHUNK_BYTES,
            "max_chunks": MAX_CHUNKS,
            "reps": args.reps,
            "timer": timer_calibration(),
            "notes": [
                "fixed-table variants are hand-kept replicas of word_cache.rs; end_to_end.on runs the real shipped code (capacity sweep only — its MAX_LENGTH is a crate constant)",
                "replay probes the cache before the ignore_merges whole-word vocab check, unlike the shipped cache — hit rates on ignore_merges models read high",
                "variant hashers use fixed seeds for reproducibility; the shipped cache seeds randomly per process",
                "naive_hashmap gets the same max_length guard as the fixed tables here; its historical v1 had none",
                "latency percentiles are raw per-op wall clock — read them against meta.timer (an op below resolution reads as 0)",
            ],
        },
        "models": models,
    });

    let rendered = serde_json::to_string_pretty(&report).unwrap();
    if let Some(out) = &args.out {
        std::fs::write(out, &rendered).unwrap();
        eprintln!("wrote {out}");
    }
    println!("{rendered}");
}
