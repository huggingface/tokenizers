#![cfg_attr(feature = "unstable", feature(iter_array_chunks))]
//! # PtrHash: Minimal Perfect Hashing at RAM Throughput
//!
//! PtrHash builds a _minimal perfect hash function_, that is,
//! a hash function that maps a fixed set of keys to `{0, ..., n-1}`.
//!
//! PtrHash was developed for large key sets of at least 1 million keys, and has been tested up to `10^11` keys.
//! In the default configuration, it uses 3.0 bits per key.
//! Nevertheless, it can also be used for arbitrary small sets.
//!
//! See the GitHub [readme](https://github.com/ragnargrootkoerkamp/ptrhash)
//! or paper ([arXiv](https://arxiv.org/abs/2502.15539), [blog version](https://curiouscoding.nl/posts/ptrhash/))
//! for details on the algorithm and performance.
//!
//! Usage example:
//! ```rust
//! use ptr_hash::PtrHashParams;
//!
//! // Enable logging.
//! env_logger::init();
//!
//! // Generate some random keys.
//! let n = 1_000_000;
//! let keys = ptr_hash::util::generate_keys(n);
//!
//! // Build the default variant of the datastructure.
//! // See `FastPtrHash` and `CompactPtrHash` below as alternatives.
//! let mphf = <ptr_hash::DefaultPtrHash>::new(&keys, PtrHashParams::default());
//!
//! // Get the index of a key.
//! let key = 0;
//! let idx = mphf.index(&key);
//! assert!(idx < n);
//!
//! // An iterator over the indices of the keys.
//! // 32: number of iterations ahead to prefetch.
//! // _: placeholder to infer the type of keys being iterated.
//! let indices = mphf.index_stream::<32, _>(&keys);
//! assert_eq!(indices.sum::<usize>(), (n * (n - 1)) / 2);
//!
//! // Query a batch of keys.
//! let query_keys = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15];
//! let mut indices = mphf.index_batch::<16, _>(query_keys);
//! indices.sort();
//! for i in 0..indices.len()-1 {
//!     assert!(indices[i] != indices[i+1]);
//! }
//!
//! // Test that all items map to different indices
//! let mut taken = vec![false; n];
//! for key in &keys {
//!     let idx = mphf.index(&key);
//!     assert!(!taken[idx]);
//!     taken[idx] = true;
//! }
//!
//! // `FastPtrHash` skips remapping and returns values up to `PtrHash::max_index()`,
//! // which is around 1.01*n.
//! let phf = <ptr_hash::FastPtrHash>::new(&keys, PtrHashParams::default());
//!
//! for key in &keys {
//!     let idx = mphf.index(&key);
//!     // NOTE: Not `n` but `phf.max_index()` here!
//!     assert!(idx < phf.max_index());
//! }
//!
//! // `CompactPtrHash` uses multi-threaded construction and is more space-efficient,
//! // but slightly slower to query.
//! let phf = <ptr_hash::CompactPtrHash>::new(&keys, PtrHashParams::default_compact());
//!
//! for key in &keys {
//!     let idx = mphf.index(&key);
//!     assert!(idx < n);
//! }
//! ```
//!
//! ## Default configurations
//!
//! - For fastest query throughput, use [`FastPtrHash`] (2.67 bits/key), which is a non-minimal PHF that skips remapping values into `[0, n)`.67 bits/key.
//! - If you want a minimal PHF that returns values in `[0, n)`, use [`DefaultPtrHash`] (3.0 bits/key).
//! - If you have many keys, use [`CompactPtrHash`] (2.15 bits/key), which allows for multi-threaded construction
//!   by splitting the input into multiple parts and uses a more space-efficient bucket function.
//!   Queries will be a bit slower because the part and index inside the part are computed separately.
//!   `PtrHashParams::default_balanced()` is smaller and still fast to construct, while
//!   `PtrHashParams::default_compact()` is even smaller and around 2x slower to construct.
//!
//! ## Hash functions
//!
//! PtrHash benefits from using an as-fast-as-possible hash function.
//!
//! - If your keys are already random integers, use [`hash::NoHash`].
//! - For integers, use [`hash::FastIntHash`], which aliases the fast-but-weak [`hash::FxHash`]. Otherwise, try [`hash::StrongerIntHash`].
//! - For strings, use [`hash::StringHash`] when the number of keys is at most `10^9`, and use [`hash::StringHash128`] for more keys. These alias [`hash::Gx`] and [`hash::Gx128`].
//!
//! See the [`hash`] module documentation for better hashes in case these cause hash collisions.
//!
//! ```
//! // Hashing strings
//! # #[cfg(feature = "gxhash")] {
//! use ptr_hash::{DefaultPtrHash, PtrHashParams, hash::StringHash};
//!
//! let keys = vec!["abc", "def"];
//! let mphf = <DefaultPtrHash<StringHash, _>>::new(&keys, PtrHashParams::default());
//!
//! let idx = mphf.index(&"def");
//! # }
//! ```
//!
//! ## Partitioning
//!
//! By default, PtrHash builds all keys as a single part.
//!
//! Faster multi-threaded construction is possible using `SINGLE_PART=false` (via [`CompactPtrHash`]),
//! which splits the keys over multiple parts.
//! Additionally, having fewer keys per part improves the cache-locality of the construction.
//! Query time is slightly slower though, since computing the part and index
//! inside the part are two separate steps.
//!
//! ## Sharding
//!
//! When the keys and/or their hashes do not all fit in memory at once, use sharding.
//! This requires `SINGLE_PART=false`, e.g. via [`CompactPtrHash`].
//! See [`shard::Sharding`] for details of different sharding methods.
//! ```
//! use ptr_hash::{CompactPtrHash, PtrHashParams, Sharding};
//!
//! let mut params = PtrHashParams::default_compact();
//! // The default value. For ~16GB of u64 hashes or ~32GB of u128 hashes.
//! // Make sure to also leave space for the data structure itself.
//! params.keys_per_shard = 1<<31;
//! params.sharding = Sharding::Disk;
//!
//! let keys = vec![1,2,3]; // 10^12 or who knows how many keys.
//! let mphf = <CompactPtrHash>::new(&keys, params);
//! ```
//!
//! ## Reducing space usage
//!
//! The default parameters are chosen for reliability, construction speed, and query speed, and give around 3 bits per keys.
//! To achieve smaller sizes, consider using [`cacheline_ef::CachelineEfVec`] or [`pack::EliasFano`] as 'remap' structure, instead of `Vec<u32>`.
//!
//! Additionally, one can use [`CompactPtrHash`] with [`PtrHashParams::default_balanced()`] parameters, which use the `CubicEps` bucket function instead of `Linear`, and increase `lambda` from the default of `3.0` to `3.5`.
//! [`PtrHashParams::default_compact()`] is even smaller, but slower to construct, and generally slightly less reliable.
//!
//! ```
//! # #[cfg(feature = "elias-fano")] {
//! use ptr_hash::{PtrHash, PtrHashParams};
//!
//! let params = PtrHashParams::default_balanced();
//! let keys = vec![1u64, 2, 3];
//! let mphf = <PtrHash<_, _, ptr_hash::pack::EliasFano>>::new(&keys, params);
//! # }
//! ```

/// Customizable Hasher trait.
pub mod hash;
/// Extendable backing storage trait and types.
pub mod pack;
/// Some internal logging and testing utilities.
pub mod util;

pub mod bucket_fn;
mod bucket_idx;
mod build;
mod fastmod;
mod reduce;
mod shard;
mod sort_buckets;
#[doc(hidden)]
pub mod stats;
#[cfg(test)]
mod test;

use bitvec::{bitvec, vec::BitVec};
use bucket_fn::BucketFn;
use bucket_fn::CubicEps;
use bucket_fn::Linear;
use bucket_fn::SquareEps;
use itertools::izip;
use itertools::Itertools;
use log::debug;
use log::trace;
use pack::MutPacked;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
pub use shard::Sharding;
use stats::BucketStats;
use std::array::from_fn;
use std::{borrow::Borrow, default::Default, marker::PhantomData, time::Instant};

use crate::{hash::*, pack::Packed, reduce::*, util::log_duration};

/// Parameters for PtrHash construction.
///
/// While all fields are public, prefer one of the default functions,
/// [`PtrHashParams::default()`], [`PtrHashParams::default_fast()`], or
/// [`PtrHashParams::default_compact()`].
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[cfg_attr(feature = "epserde", epserde(deep_copy))]
pub struct PtrHashParams<BF> {
    /// Use `n/alpha` slots approximately.
    pub alpha: f64,
    /// Use average bucket size lambda.
    pub lambda: f64,
    /// Bucket function
    pub bucket_fn: BF,
    /// Upper bound on number of keys per shard.
    /// Default is 2^31, or 16GB of u64 hashes per shard.
    pub keys_per_shard: usize,
    /// When true, write each shard to a file instead of iterating multiple
    /// times.
    pub sharding: Sharding,
}

impl PtrHashParams<Linear> {
    /// Parameters for fast construction and queries. Use these by default.
    ///
    /// Takes `3.0` bits/key, and can be up to 2x faster to query than the balanced or compact versions.
    /// - `alpha=0.99`
    /// - `lambda=3.0`
    /// - `bucket_fn=Linear`
    ///
    pub fn default_fast() -> Self {
        Self {
            alpha: 0.99,
            lambda: 3.0,
            bucket_fn: Linear,
            keys_per_shard: 1 << 31,
            sharding: Sharding::None,
        }
    }
}

#[doc(hidden)]
impl PtrHashParams<SquareEps> {
    pub fn default_square() -> Self {
        Self {
            alpha: 0.99,
            lambda: 3.5,
            bucket_fn: SquareEps,
            keys_per_shard: 1 << 31,
            sharding: Sharding::None,
        }
    }
}

impl PtrHashParams<CubicEps> {
    /// Balanced parameters, which saves some space for larger inputs.
    /// This is the 'Default' from the paper.
    ///
    /// Takes `2.4` bits/key, and trades off space and speed.
    /// - `alpha=0.99`
    /// - `lambda=3.5`
    /// - `bucket_fn=CubicEps`
    pub fn default_balanced() -> Self {
        Self {
            alpha: 0.99,
            lambda: 3.5,
            bucket_fn: CubicEps,
            keys_per_shard: 1 << 31,
            sharding: Sharding::None,
        }
    }

    /// Default 'compact' parameters.
    ///
    /// Takes `2.15` bits/key, but is typically 2x slower to construct than the default version.
    /// If construction fails, try again with decreased `lambda`.
    /// - `alpha=0.99`
    /// - `lambda=3.9`
    /// - `bucket_fn=CubicEps`
    pub fn default_compact() -> Self {
        Self {
            alpha: 0.99,
            lambda: 3.9,
            bucket_fn: CubicEps,
            keys_per_shard: 1 << 31,
            sharding: Sharding::None,
        }
    }
}

/// By default, use [`PtrHashParams::default_fast()`].
impl Default for PtrHashParams<Linear> {
    fn default() -> Self {
        Self::default_fast()
    }
}

/// Fastest variant: a **non-minimal** PHF that **may return values slightly larger than n**. 2.67 bits/key.
///
/// This skips remapping values into `[0, n)`, and instead returns values up to [`PtrHash::max_index()`].
///
/// Use as [`FastPtrHash::<Hasher, _>::new(&keys, PtrHashParams::default_fast())`].
pub type FastPtrHash<Hx = hash::FastIntHash, Key = u64> =
    PtrHash<Key, bucket_fn::Linear, Vec<u32>, Hx, Vec<u8>, true, false>;

/// Default variant: a minimal PHF that returns values in `[0, n)`. 3.0 bits/key.
///
/// Use as [`<DefaultPtrHash<Hasher>>::new(&keys, PtrHashParams::default())`].
pub type DefaultPtrHash<Hx = hash::FastIntHash, Key = u64> =
    PtrHash<Key, bucket_fn::Linear, Vec<u32>, Hx, Vec<u8>, true, true>;

/// Variant for large data sets with multi-threaded construction. 2.15 bits/key.
///
/// This version optimizes space usage rathen than query speed:
/// - Uses multiple parts to allow multi-threaded construction.
/// - Uses the `CubicEps` bucket function to reduce space usage.
/// - Remaps keys into `[0, n)` to achieve a minimal PHF.
///
/// This uses 2.15 bits/key with EliasFano, and 2.35 bits/key otherwise.
///
/// Use as [`<CompactPtrHash<Hasher>>::new(&keys, PtrHashParams::default_compact())`].
#[cfg(feature = "elias-fano")]
pub type CompactPtrHash<Hx = hash::FastIntHash, Key = u64> =
    PtrHash<Key, bucket_fn::CubicEps, pack::EliasFano, Hx, Vec<u8>, false, true>;
#[cfg(not(feature = "elias-fano"))]
pub type CompactPtrHash<Hx = hash::FastIntHash, Key = u64> =
    PtrHash<Key, bucket_fn::CubicEps, Vec<u32>, Hx, Vec<u8>, false, true>;

/// Trait that keys must satisfy.
pub trait KeyT: Send + Sync + std::hash::Hash {}
impl<T: Send + Sync + std::hash::Hash + ?Sized> KeyT for T {}

// Some fixed algorithmic decisions.
type Rp = FastReduce;
type Rb = FastReduce;
// NOTE: This is not a faithful modulo for 64-bit values, but either way only returns values up to `d`.
type RemSlots = fastmod::FM32;
type Pilot = u64;
type PilotHash = u64;

/// PtrHash datastructure.
/// It is recommended to use PtrHash with default type aliases:
/// - [`FastPtrHash`]: *non-minimal* PHF that skips remapping values into `[0, n)`.
/// - [`DefaultPtrHash`]: MPHF that returns values in `[0, n)`.
/// - [`CompactPtrHash`]: MPHF that allows for multi-threaded construction.
///
/// - `Key`: The type of keys to hash.
/// - `BF`: The bucket function to use. Inferred from `PtrHashParams` when calling `PtrHash::new()`.
/// - `F`: The packing to use for remapping free slots, default `CachelineEf`.
/// - `Hx`: The hasher to use for keys, default `FxHash` for integers, but consider
///       `hash::StringHash` (using `gxhash`) for strings, or `hash::StringHash128` when the number of string keys is very
///       large.
/// - `V`: The pilots type. Usually `Vec<u8>`, or `&[u8]` for Epserde.
/// - `SINGLE_PART`: using a single part gives faster queries, but prevents multi-threaded construction (default: true).
/// - `REMAP`: remapping results in a minimal PHF, but is slightly slower (default: true).
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[derive(Clone)]
pub struct PtrHash<
    Key: KeyT + ?Sized = u64,
    BF: BucketFn = bucket_fn::Linear,
    F: Packed = Vec<u32>,
    Hx: KeyHasher<Key> = hash::FastIntHash,
    V: AsRef<[u8]> = Vec<u8>,
    const SINGLE_PART: bool = true,
    const REMAP: bool = true,
> {
    params: PtrHashParams<BF>,

    /// The number of keys.
    n: usize,
    /// The total number of parts.
    parts: usize,
    /// The number of shards.
    shards: usize,
    /// The maximal number of parts per shard.
    /// The last shard may have fewer parts.
    parts_per_shard: usize,
    /// The total number of slots.
    slots_total: usize,
    /// The total number of buckets.
    buckets_total: usize,
    /// The number of slots per part, always a power of 2.
    slots: usize,
    /// The number of buckets per part.
    buckets: usize,

    // Precomputed fast modulo operations.
    /// Fast %shards.
    rem_shards: Rp,
    /// Fast %parts.
    rem_parts: Rp,
    /// Fast %b.
    rem_buckets: Rb,
    /// Fast %b_total.
    rem_buckets_total: Rb,

    /// Fast %s when there is only a single part.
    rem_slots: RemSlots,

    // Computed state.
    /// The global seed.
    seed: u64,
    /// The pilots.
    pilots: V,
    /// Remap the out-of-bound slots to free slots.
    remap: F,
    _key: PhantomData<Key>,
    _hx: PhantomData<Hx>,
}

/// An empty PtrHash instance. Mostly useless, but may be convenient.
impl<
        Key: KeyT,
        BF: BucketFn,
        F: MutPacked,
        Hx: KeyHasher<Key>,
        const SINGLE_PART: bool,
        const REMAP: bool,
    > Default for PtrHash<Key, BF, F, Hx, Vec<u8>, SINGLE_PART, REMAP>
where
    PtrHashParams<BF>: Default,
{
    fn default() -> Self {
        PtrHash {
            params: <PtrHashParams<BF> as Default>::default(),

            n: 0,
            parts: 0,
            shards: 0,
            parts_per_shard: 0,
            slots_total: 0,
            buckets_total: 0,
            slots: 0,
            buckets: 0,
            rem_shards: FastReduce::new(0),
            rem_parts: FastReduce::new(0),
            rem_buckets: FastReduce::new(0),
            rem_buckets_total: FastReduce::new(0),
            rem_slots: RemSlots::new(0),
            seed: 0,
            pilots: vec![],
            remap: F::default(),
            _key: PhantomData,
            _hx: PhantomData,
        }
    }
}

/// Construction methods taking a list of keys.
impl<
        Key: KeyT,
        BF: BucketFn,
        F: MutPacked,
        Hx: KeyHasher<Key>,
        const SINGLE_PART: bool,
        const REMAP: bool,
    > PtrHash<Key, BF, F, Hx, Vec<u8>, SINGLE_PART, REMAP>
{
    /// Create a new PtrHash instance from the given keys.
    ///
    /// Use `<PtrHash>::new()` or `DefaultPtrHash::new()` instead of simply `PtrHash::new()` to
    /// get the default values for the generics.
    ///
    /// NOTE: This panics when construction fails after 10 attempts.
    /// This should be rare, but can happen if we are unlucky with the rng seeds.
    /// Consider calling [`PtrHash::try_new()`] instead.
    ///
    /// NOTE: Only up to 2^40 keys are supported.
    pub fn new(keys: &[Key], params: PtrHashParams<BF>) -> Self {
        let mut ptr_hash = Self::init(keys.len(), params);
        ptr_hash
            .compute_pilots(keys.par_iter())
            .expect("Unable to construct PtrHash after 10 tries. Try using a better hash or decreasing lambda.");
        ptr_hash
    }

    /// Version that returns build statistics.
    #[doc(hidden)]
    pub fn new_with_stats(keys: &[Key], params: PtrHashParams<BF>) -> (Self, BucketStats) {
        let mut ptr_hash = Self::init(keys.len(), params);
        let stats = ptr_hash
            .compute_pilots(keys.par_iter())
            .expect("Unable to construct PtrHash after 10 tries. Try using a better hash or decreasing lambda.");
        (ptr_hash, stats)
    }

    /// Fallible version of `new` that returns `None` if construction fails.
    /// This can happen when `lambda` is too larger (e.g. for `default_compact`
    /// parameters) and the eviction chains become too long.
    pub fn try_new(keys: &[Key], params: PtrHashParams<BF>) -> Option<Self> {
        let mut ptr_hash = Self::init(keys.len(), params);
        ptr_hash.compute_pilots(keys.par_iter())?;
        Some(ptr_hash)
    }
}

/// Construction (helper) methods working with unsized keys.
impl<
        Key: KeyT + ?Sized,
        BF: BucketFn,
        F: MutPacked,
        Hx: KeyHasher<Key>,
        const SINGLE_PART: bool,
        const REMAP: bool,
    > PtrHash<Key, BF, F, Hx, Vec<u8>, SINGLE_PART, REMAP>
{
    /// Same as `new` above, but takes a `ParallelIterator` over keys instead of a slice.
    ///
    /// The iterator must be cloneable, since construction can fail for the
    /// first seed (e.g. due to duplicate hashes), in which case a new pass over
    /// keys is need.
    pub fn new_from_par_iter<'a>(
        n: usize,
        keys: impl ParallelIterator<Item = impl Borrow<Key>> + Clone + 'a,
        params: PtrHashParams<BF>,
    ) -> Self {
        let mut ptr_hash = Self::init(n, params);
        ptr_hash.compute_pilots(keys);
        ptr_hash
    }

    /// Only initialize the parameters; do not compute the pilots yet.
    fn init(n: usize, mut params: PtrHashParams<BF>) -> Self {
        // assert!(n < (1 << 40), "Number of keys must be less than 2^40.");

        let shards = match (SINGLE_PART, params.sharding) {
            (_, Sharding::None) => 1,
            (true, _) => panic!("Sharding does not work in combination with SINGLE_PART=true."),
            _ => n.div_ceil(params.keys_per_shard),
        };

        // Formula of Vigna, eps-cost-sharding: https://arxiv.org/abs/2503.18397
        // (1-alpha)/2, so that on average we still have some room to play with.
        let parts = if SINGLE_PART {
            1
        } else {
            let eps = (1.0 - params.alpha) / 2.0;
            let x = n as f64 * eps * eps / 2.0;
            // Half the number of target parts for some safety margin.
            // Otherwise, the largest part is often still too large for small inputs.
            let target_parts = x / x.ln() / 2.0;
            let parts_per_shard = (target_parts.floor() as usize) / shards;
            parts_per_shard.max(1) * shards
        };

        let keys_per_part = n / parts;
        let parts_per_shard = parts / shards;
        let mut slots_per_part = (keys_per_part as f64 / params.alpha) as usize;
        // Avoid powers of two, since then %S does not depend on all bits.
        if slots_per_part.is_power_of_two() {
            slots_per_part += 1;
        }
        let slots_total = parts * slots_per_part;
        // Add a few extra buckets to avoid collisions for small n.
        let buckets_per_part = (keys_per_part as f64 / params.lambda).ceil() as usize + 3;
        let buckets_total = parts * buckets_per_part;

        trace!("        keys: {n:>10}");
        trace!("      shards: {shards:>10}");
        trace!("       parts: {parts:>10}");
        trace!("   slots/prt: {slots_per_part:>10}");
        trace!("   slots tot: {slots_total:>10}");
        trace!("  real alpha: {:>10.4}", n as f64 / slots_total as f64);
        trace!(" buckets/prt: {buckets_per_part:>10}");
        trace!(" buckets tot: {buckets_total:>10}");
        trace!("keys/ bucket: {:>13.2}", n as f64 / buckets_total as f64);

        params
            .bucket_fn
            .set_buckets_per_part(buckets_per_part as u64);

        Self {
            params,
            n,
            parts,
            shards,
            parts_per_shard,
            slots_total,
            slots: slots_per_part,
            buckets_total,
            buckets: buckets_per_part,
            rem_shards: Rp::new(shards),
            rem_parts: Rp::new(parts),
            rem_buckets: Rb::new(buckets_per_part),
            rem_buckets_total: Rb::new(buckets_total),
            rem_slots: RemSlots::new(slots_per_part.max(1)), // fix for n=0
            seed: 0,
            pilots: Default::default(),
            remap: F::default(),
            _key: PhantomData,
            _hx: PhantomData,
        }
    }

    fn compute_pilots<'a>(
        &mut self,
        keys: impl ParallelIterator<Item = impl Borrow<Key>> + Clone + 'a,
    ) -> Option<BucketStats> {
        let overall_start = std::time::Instant::now();
        // Initialize arrays;
        let mut taken: Vec<BitVec> = vec![];
        let mut pilots: Vec<u8> = vec![];

        let mut tries = 0;
        const MAX_TRIES: usize = 10;

        let mut rng = ChaCha8Rng::seed_from_u64(31415);

        // Loop over global seeds `s`.
        let stats = 's: loop {
            tries += 1;
            if tries > MAX_TRIES {
                log::error!("PtrHash failed to find a global seed after {MAX_TRIES} tries.");
                return None;
            }

            let old_seed = self.seed;

            // Choose a global seed s.
            self.seed = rng.random();
            if tries == 1 {
                log::trace!("First seed tried: {}", self.seed);
            } else {
                log::warn!("Previous seed {old_seed} failed.");
                log::warn!("Trying seed number {tries}: {}.", self.seed);
            }

            // Reset output-memory.
            pilots.clear();
            pilots.resize(self.buckets_total, 0);

            // TODO: Compress taken on the fly, instead of pre-allocating the entire thing.
            for taken in taken.iter_mut() {
                taken.clear();
                taken.resize(self.slots, false);
            }
            taken.resize_with(self.parts, || bitvec![0; self.slots]);

            // Iterate over shards.
            let shard_hashes = self.shards(keys.clone());
            // Avoid chunks_mut(0) when n=0.
            let shard_pilots = pilots.chunks_mut((self.buckets * self.parts_per_shard).max(1));
            let shard_taken = taken.chunks_mut(self.parts_per_shard);
            let mut stats = BucketStats::default();
            // eprintln!("Num shards (keys) {}", shard_keys.());
            for (shard, (hashes, pilots, taken)) in
                izip!(shard_hashes, shard_pilots, shard_taken).enumerate()
            {
                // Determine the buckets.
                let start = std::time::Instant::now();
                let Some((hashes, part_starts)) = self.sort_parts(shard, hashes) else {
                    trace!("Found duplicate hashes");
                    // Found duplicate hashes.
                    continue 's;
                };
                let start = log_duration("sort buckets", start);

                // Compute pilots.
                if let Some(shard_stats) =
                    self.build_shard(shard, &hashes, &part_starts, pilots, taken)
                {
                    stats.merge(shard_stats);
                    log_duration("find pilots", start);
                } else {
                    trace!("Could not find pilots");
                    continue 's;
                }
            }

            let start = std::time::Instant::now();
            let remap = self.remap_free_slots(&taken);
            log_duration("remap free", start);
            if remap.is_err() {
                trace!("Failed to construct CachelineEF");
                continue 's;
            }

            break 's stats;
        };

        // Pack the data.
        self.pilots = pilots;

        let (p, r) = self.bits_per_element();
        debug!("bits/element: {}", p + r);
        log_duration("total build", overall_start);
        Some(stats)
    }

    fn remap_free_slots(&mut self, taken: &Vec<BitVec>) -> Result<(), ()> {
        assert_eq!(
            taken.iter().map(|t| t.count_zeros()).sum::<usize>(),
            self.slots_total - self.n,
            "Not the right number of free slots left!\n total slots {} - n {}",
            self.slots_total,
            self.n
        );

        if !REMAP || self.slots_total == self.n {
            return Ok(());
        }

        // Compute the free spots.
        let mut v = Vec::with_capacity(self.slots_total - self.n);
        let get = |t: &Vec<BitVec>, idx: usize| t[idx / self.slots][idx % self.slots];
        for i in taken
            .iter()
            .enumerate()
            .flat_map(|(p, t)| {
                let offset = p * self.slots;
                t.iter_zeros().map(move |i| offset + i)
            })
            .take_while(|&i| i < self.n)
        {
            while !get(&taken, self.n + v.len()) {
                v.push(i as u64);
            }
            v.push(i as u64);
        }
        self.remap = MutPacked::try_new(v).ok_or(())?;
        Ok(())
    }
}

/// Indexing methods.
impl<
        Key: KeyT + ?Sized,
        BF: BucketFn,
        F: Packed,
        Hx: KeyHasher<Key>,
        V: AsRef<[u8]>,
        const SINGLE_PART: bool,
        const REMAP: bool,
    > PtrHash<Key, BF, F, Hx, V, SINGLE_PART, REMAP>
{
    /// Return the number of bits per element used for the pilots (`.0`) and the
    /// remapping (`.1`).
    pub fn bits_per_element(&self) -> (f64, f64) {
        let pilots = self.pilots.as_ref().size_in_bytes() as f64 / self.n as f64;
        let remap = self.remap.size_in_bytes() as f64 / self.n as f64;
        (8. * pilots, 8. * remap)
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// `self.index()` always returns below this bound.
    /// Around `n/alpha ~ 1.01*n` when `REMAP=false`, or `n` when `REMAP=true`.
    pub fn max_index(&self) -> usize {
        if REMAP {
            self.n
        } else {
            self.slots_total
        }
    }

    pub fn slots_per_part(&self) -> usize {
        self.slots
    }

    /// Get the index for `key`.
    ///
    /// When `REMAP` is true, returns an index in `[0, n)`.
    /// When `REMAP` is false, returns a non-minimal index in `[0, self.max_index())`.
    ///
    /// When `SINGLE_PART` is true, this uses slightly faster single-part hashing.
    #[inline(always)]
    pub fn index(&self, key: &Key) -> usize {
        let slot = self.index_no_remap(key);

        if REMAP {
            if slot < self.n {
                slot
            } else {
                self.remap.index(slot - self.n) as usize
            }
        } else {
            slot
        }
    }

    /// Get the non-remapped index for `key`, regardless of `REMAP`.
    ///
    /// When `SINGLE_PART` is true, uses slightly faster single-part hashing.
    #[doc(hidden)]
    #[inline(always)]
    pub fn index_no_remap(&self, key: &Key) -> usize {
        if SINGLE_PART {
            debug_assert_eq!(self.parts, 1);
        }

        let hx = self.hash_key(key);
        let slot = if SINGLE_PART {
            let b = self.bucket_in_part(hx.high());
            let pilot = self.pilots.as_ref().index(b);
            self.slot_in_part(hx, pilot)
        } else {
            let b = self.bucket(hx);
            let pilot = self.pilots.as_ref().index(b);
            self.slot(hx, pilot)
        };

        slot
    }

    /// Takes an iterator over keys and returns an iterator over the indices of the keys.
    ///
    /// Uses a buffer of size `B` for prefetching ahead. `B=32` should be a good choice.
    /// When `REMAP` is true (the default), returns minimal indices in `[0, n)`.
    /// When `REMAP` is false, returns non-minimal indices in `[0, n/alpha)`.
    /// The iterator can return either `Q=Key` or `Q=&Key`.
    ///
    /// See the module-level documentation for an example.
    // NOTE: It would be cool to use SIMD to determine buckets/positions in
    // parallel, but this is complicated, since SIMD doesn't support the
    // 64x64->128 multiplications needed in bucket/slot computations.
    #[inline]
    pub fn index_stream<'a, const B: usize, Q: Borrow<Key> + 'a>(
        &'a self,
        keys: impl IntoIterator<Item = Q> + 'a,
    ) -> impl Iterator<Item = usize> + 'a {
        self.index_stream_maybe_remap::<B, REMAP, _>(keys)
    }

    /// Version that allows disabling remapping.
    #[doc(hidden)]
    #[inline]
    pub fn index_stream_maybe_remap<
        'a,
        const B: usize,
        const QUERY_REMAP: bool,
        Q: Borrow<Key> + 'a,
    >(
        &'a self,
        keys: impl IntoIterator<Item = Q> + 'a,
    ) -> impl Iterator<Item = usize> + 'a {
        let mut keys = keys.into_iter();

        // Ring buffers to cache the hash and bucket of upcoming queries.
        let mut next_hashes: [Hx::H; B] = [Hx::H::default(); B];
        let mut next_buckets: [usize; B] = [0; B];

        // Initialize and prefetch first B values.
        let mut leftover = B;
        for idx in 0..B {
            let hx = keys
                .next()
                .map(|k| {
                    leftover -= 1;
                    self.hash_key(k.borrow())
                })
                .unwrap_or_default();
            next_hashes[idx] = hx;

            next_buckets[idx] = self.bucket(next_hashes[idx]);
            prefetch_index::prefetch_index(self.pilots.as_ref(), next_buckets[idx]);
        }

        // Manual iterator implementation so we avoid the overhead and
        // non-inlining of Chain, and instead have a manual fold.
        struct It<
            'a,
            const B: usize,
            Key: KeyT + ?Sized,
            Q: Borrow<Key> + 'a,
            KeyIt: Iterator<Item = Q> + 'a,
            BF: BucketFn,
            F: Packed,
            Hx: KeyHasher<Key>,
            V: AsRef<[u8]>,
            const SINGLE_PART: bool,
            const REMAP: bool,
            const QUERY_REMAP: bool,
        > {
            ph: &'a PtrHash<Key, BF, F, Hx, V, SINGLE_PART, REMAP>,
            keys: KeyIt,
            next_hashes: [Hx::H; B],
            next_buckets: [usize; B],
            leftover: usize,
        }

        impl<
                'a,
                const B: usize,
                Key: KeyT + ?Sized,
                Q: Borrow<Key> + 'a,
                KeyIt: Iterator<Item = Q> + 'a,
                BF: BucketFn,
                F: Packed,
                Hx: KeyHasher<Key>,
                V: AsRef<[u8]>,
                const SINGLE_PART: bool,
                const REMAP: bool,
                const QUERY_REMAP: bool,
            > Iterator for It<'a, B, Key, Q, KeyIt, BF, F, Hx, V, SINGLE_PART, REMAP, QUERY_REMAP>
        {
            type Item = usize;
            fn next(&mut self) -> Option<usize> {
                unimplemented!("Use a method that calls `fold()` instead.");
            }

            #[inline(always)]
            fn fold<BB, FF>(mut self, init: BB, mut f: FF) -> BB
            where
                Self: Sized,
                FF: FnMut(BB, Self::Item) -> BB,
            {
                let mut accum = init;
                let mut i = 0;

                for key in self.keys {
                    let next_hash = self.ph.hash_key(key.borrow());
                    let idx = i % B;
                    let cur_hash = self.next_hashes[idx];
                    let cur_bucket = self.next_buckets[idx];
                    self.next_hashes[idx] = next_hash;
                    self.next_buckets[idx] = self.ph.bucket(self.next_hashes[idx]);
                    prefetch_index::prefetch_index(self.ph.pilots.as_ref(), self.next_buckets[idx]);
                    let pilot = self.ph.pilots.as_ref().index(cur_bucket);
                    let slot = self.ph.slot(cur_hash, pilot);

                    let slot = if QUERY_REMAP && slot >= self.ph.n {
                        self.ph.remap.index(slot - self.ph.n) as usize
                    } else {
                        slot
                    };

                    accum = f(accum, slot);
                    i += 1;
                }

                for _ in 0..B - self.leftover {
                    let idx = i % B;
                    let cur_hash = self.next_hashes[idx];
                    let cur_bucket = self.next_buckets[idx];
                    let pilot = self.ph.pilots.as_ref().index(cur_bucket);
                    let slot = self.ph.slot(cur_hash, pilot);

                    let slot = if REMAP && slot >= self.ph.n {
                        self.ph.remap.index(slot - self.ph.n) as usize
                    } else {
                        slot
                    };

                    accum = f(accum, slot);
                    i += 1;
                }

                accum
            }
        }
        It::<B, _, _, _, _, _, _, _, SINGLE_PART, REMAP, QUERY_REMAP> {
            ph: self,
            keys,
            next_hashes,
            next_buckets,
            leftover,
        }
    }

    /// Query a batch of `K` keys at once.
    ///
    /// Input can be either `[Key; K]` or `[&Key; K]`.
    #[inline]
    pub fn index_batch<'a, const K: usize, Q: Borrow<Key> + 'a>(
        &'a self,
        xs: [Q; K],
    ) -> [usize; K] {
        let hashes = xs.map(|x| self.hash_key(x.borrow()));
        let mut buckets: [usize; K] = [0; K];

        // Prefetch.
        for idx in 0..K {
            buckets[idx] = self.bucket(hashes[idx]);
            prefetch_index::prefetch_index(self.pilots.as_ref(), buckets[idx]);
        }
        // Query.
        from_fn(
            #[inline(always)]
            move |idx| {
                let pilot = self.pilots.as_ref().index(buckets[idx]);
                let slot = self.slot(hashes[idx], pilot);
                if REMAP && slot >= self.n {
                    self.remap.index(slot - self.n) as usize
                } else {
                    slot
                }
            },
        )
    }

    /// Takes an iterator over keys and returns an iterator over the indices of the keys.
    ///
    /// Queries in batches of size K.
    ///
    /// NOTE: Does not process the remainder
    #[doc(hidden)]
    #[cfg(feature = "unstable")]
    #[inline]
    pub fn index_batch_exact<'a, const K: usize>(
        &'a self,
        xs: impl IntoIterator<Item = &'a Key> + 'a,
    ) -> impl Iterator<Item = usize> + 'a {
        let mut buckets: [usize; K] = [0; K];

        // Work on chunks of size K.
        let mut f = {
            #[inline(always)]
            move |hx: [Hx::H; K]| {
                // Prefetch.
                for idx in 0..K {
                    buckets[idx] = self.bucket(hx[idx]);
                    crate::util::prefetch_index(self.pilots.as_ref(), buckets[idx]);
                }
                // Query.
                (0..K).map(
                    #[inline(always)]
                    move |idx| {
                        let pilot = self.pilots.as_ref().index(buckets[idx]);
                        let slot = self.slot(hx[idx], pilot);
                        if REMAP && slot >= self.n {
                            self.remap.index(slot - self.n) as usize
                        } else {
                            slot
                        }
                    },
                )
            }
        };
        let array_chunks = xs.into_iter().map(|x| self.hash_key(x)).array_chunks::<K>();
        array_chunks.into_iter().flat_map(
            #[inline(always)]
            move |chunk| f(chunk),
        )
        // .chain(f(&array_chunks
        //     .into_remainder()
        //     .unwrap_or_default()
        //     .into_iter()))
    }

    /// A variant of index_batch_exact that scales better with K.
    /// Somehow the version above has pretty constant speed regardless of K.
    #[doc(hidden)]
    #[inline]
    pub fn index_batch_exact2<'a, const K: usize, const QUERY_REMAP: bool>(
        &'a self,
        xs: impl IntoIterator<Item = &'a Key, IntoIter: ExactSizeIterator> + 'a,
    ) -> impl Iterator<Item = usize> + 'a {
        let mut buckets: [usize; K] = [0; K];
        let mut hs: [Hx::H; K] = [Hx::H::default(); K];

        let mut xs = xs
            .into_iter()
            .map(|x| self.hash_key(x))
            .chain([Default::default(); K]);
        for i in 0..K {
            hs[i] = xs.next().unwrap();
        }
        let mut idx = K;
        xs.map(move |hx| {
            if idx == K {
                idx = 0;
                // Prefetch.
                for idx in 0..K {
                    buckets[idx] = self.bucket(hs[idx]);
                    prefetch_index::prefetch_index(self.pilots.as_ref(), buckets[idx]);
                }
            }

            // Query.
            let pilot = self.pilots.as_ref().index(buckets[idx]);
            let slot = self.slot(hs[idx], pilot);

            // Update hash in current pos and increment.
            hs[idx] = hx;
            idx += 1;

            // Remap?
            if QUERY_REMAP && slot >= self.n {
                self.remap.index(slot - self.n) as usize
            } else {
                slot
            }
        })
    }

    fn hash_key(&self, x: &Key) -> Hx::H {
        Hx::hash(x, self.seed)
    }

    fn hash_pilot(&self, p: Pilot) -> PilotHash {
        hash::C.wrapping_mul(p ^ self.seed)
    }

    fn shard(&self, hx: Hx::H) -> usize {
        self.rem_shards.reduce(hx.high())
    }

    fn part(&self, hx: Hx::H) -> usize {
        self.rem_parts.reduce(hx.high())
    }

    /// Map `hx_remainder` to a bucket in the range [0, self.b).
    /// Hashes <self.p1 are mapped to large buckets [0, self.p2).
    /// Hashes >=self.p1 are mapped to small [self.p2, self.b).
    ///
    /// (Unless SPLIT_BUCKETS is false, in which case all hashes are mapped to [0, self.b).)
    fn bucket_in_part(&self, x: u64) -> usize {
        if BF::LINEAR {
            self.rem_buckets.reduce(x)
        } else if BF::B_OUTPUT {
            self.params.bucket_fn.call(x) as usize
        } else {
            self.rem_buckets.reduce(self.params.bucket_fn.call(x))
        }
    }

    /// See bucket.rs for additional implementations.
    /// Returns the offset in the slots array for the current part and the bucket index.
    fn bucket(&self, hx: Hx::H) -> usize {
        if BF::LINEAR {
            return self.rem_buckets_total.reduce(hx.high());
        }

        // Extract the high bits for part selection; do normal bucket
        // computation within the part using the remaining bits.
        // NOTE: This is somewhat slow, but doing better is hard.
        let (part, hx) = self.rem_parts.reduce_with_remainder(hx.high());
        let bucket = self.bucket_in_part(hx);
        part * self.buckets + bucket
    }

    /// Slot uses the 64 low bits of the hash.
    fn slot(&self, hx: Hx::H, pilot: u64) -> usize {
        (self.part(hx) * self.slots) + self.slot_in_part(hx, pilot)
    }

    fn slot_in_part(&self, hx: Hx::H, pilot: Pilot) -> usize {
        self.slot_in_part_hp(hx, self.hash_pilot(pilot))
    }

    /// Slot uses the 64 low bits of the hash.
    fn slot_in_part_hp(&self, hx: Hx::H, hp: PilotHash) -> usize {
        self.rem_slots.reduce(hx.low() ^ hp)
    }
}
