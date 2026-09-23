# PtrHash: Minimal Perfect Hashing at RAM Throughput

[![crates.io](https://img.shields.io/crates/v/ptr_hash.svg)](https://crates.io/crates/ptr_hash)
[![docs.rs](https://img.shields.io/docsrs/ptr_hash.svg)](https://docs.rs/ptr_hash)

PtrHash is a fast and space efficient *minimal perfect hash function* that maps
a list of `n` distinct keys into `{0,...,n-1}`.
It is based on/inspired by [PTHash](https://github.com/jermp/pthash) (and much
more than just a Rust rewrite).

**Paper.**

*Ragnar Groot Koerkamp*. PtrHash: Minimal Perfect Hashing at RAM Throughput.
SEA (2025). [doi.org/10.4230/LIPIcs.SEA.2025.21](https://doi.org/10.4230/LIPIcs.SEA.2025.21)

**Evals.** Source code for the paper evals can be found in
[examples/evals.rs](examples/evals.rs), and analysis is [evals.py](evals.py).
Plots can be found [in the blog](https://github.com/RagnarGrootKoerkamp/research/blob/master/posts/ptrhash/).
The paper evals were done on the `evals` branch (which is v1.0 with GxHash added
for string hashing) and my [fork](https://github.com/ragnargrootkoerkamp/MPHF-Experiments) of [mphf-experiments](https://github.com/ByteHamster/MPHF-Experiments).

For changes since then, see [CHANGELOG.md](./CHANGELOG.md).

**Contact.**

In case you run into any kind of issue or things are unclear,
please make issues and/or PRs, or reach out on [twitter]((https://twitter.com/curious_coding))/[bsky](https://bsky.app/profile/curiouscoding.nl).
I'm more than happy to help out with integrating PtrHash.

## Performance on small inputs

Space usage and query throughput of a for-loop in ns/key as measured by `examples/query_bench.rs`:

| method         | bits/key | hash   | 100k | 1M  | 10M | 100 M |
|----------------|----------|--------|------|-----|-----|-------|
| FastPtrHash    | 2.67     | NoHash | 1.5  | 2.2 | 3.1 | 7.6   |
|                |          | FxHash | 1.7  | 2.5 | 3.2 | 8.2   |
| DefaultPtrHash | 3.00     | NoHash | 1.7  | 2.6 | 3.3 | 8.5   |
|                |          | FxHash | 2.0  | 2.7 | 3.7 | 9.1   |
| CompactPtrHash | 2.15     | NoHash | 4.4  | 5.5 | 6.8 | 15.6  |
|                |          | FxHash | 4.8  | 5.8 | 7.2 | 15.7  |


## Performance on large input

PtrHash supports up to `2^40` keys (and probably more).
For `n=10^9` integer keys with `FxHash`, we get the following on my `i7-10750H`
at `3.6GHz` with 6 cores (see `examples/large_bench.rs`).
The construction uses multi-threading only for `CompactPtrHash`, and remapping
is skipped for `FastPtrHash`.
The last two columns indicate the query throughput for streaming queries with
prefetching 32 iterations ahead, and for prefetching in batches of 32.

| method         | pilots bits/key | remap  bits/key | space bits/key | construction ns/key | loop ns/key | stream ns/key | batch ns/key |
|----------------|-----------------|-----------------|----------------|---------------------|-------------|---------------|--------------|
| FastPtrHash    | 2.67            | -               | 2.67           | 343.5               | 9.7         | 7.5           | 9.0          |
| DefaultPtrHash | 2.67            | 0.33            | 3.00           | 349.9               | 12.0        | 8.3           | 9.4          |
| CompactPtrHash | 2.05            | 0.10            | 2.15           | 49.0 (12t)          | 19.8        | 8.5           | 10.1         |

Streaming query throughput per thread fully saturates the memory bandwidth of each
core (around 7.5 ns/cache line), and with multi-threading the full DDR4 memory bandwidth is saturated.

## Input

PtrHash is primarily intended to be used on large sets of keys, say of size at
least 1 million. Nevertheless, it can also be used for sets as small as e.g. 10
keys. In this case, there will be a relatively large constant space overhead,
and other methods may be smaller and/or faster.
(PtrHash should work fine and be reasonably fast, but for such small inputs the space-efficient
design of PtrHash makes little sense and faster queries might be possible.)


## Usage

See [docs.rs](https://docs.rs/ptr_hash) for the different variants and parameters.
Below, we use `PtrHashParams::default()` for a reasonable trade-off between size
(3.0 bits/key) and speed.
Slightly smaller size (2.15 bits/key) is possible using `CompactPtrHash` with `PtrHashParams::default_compact()`,
at the cost of significantly slower construction time (2x) and lowered reliability.

There is also `PtrHashParams::default_fast()`, which takes 2.67 bits/key but
can be almost 2x faster than the compact variant when querying integer keys in tight loops.
For large inputs, maximum query throughput is achieved with `index_stream` with default parameters.

```rust
use ptr_hash::{PtrHash, PtrHashParams};

// Generate some random keys.
let n = 1_000_000_000;
let keys = ptr_hash::util::generate_keys(n);

// Build the datastructure.
let mphf = <DefaultPtrHash>::new(&keys, PtrHashParams::default());

// Get the minimal index of a key.
let key = 0;
let idx = mphf.index(&key);
assert!(idx < n);

// Get the non-minimal index of a key. Slightly faster, but can be >=n.
let _idx = mphf.index(&key);

// An iterator over the indices of the keys.
// 32: number of iterations ahead to prefetch.
// true: remap to a minimal key in [0, n).
let indices = mphf.index_stream::<32, _>(&keys);
assert_eq!(indices.sum::<usize>(), (n * (n - 1)) / 2);

// Test that all items map to different indices
let mut taken = vec![false; n];
for key in &keys {
    let idx = mphf.index(&key);
    assert!(!taken[idx]);
    taken[idx] = true;
}

// In case you want maximum query throughput at the cost of returning non-minimal values,
// use `FastPtrHash`. `phf.max_index()` will be roughly `1.01*n`.
let phf = <FastPtrHash>::new(&keys, PtrHashParams::default());

for key in &keys {
    let idx = phf.index(&key);
    assert!(idx < phf.max_index());
}

// To enable parallel construction and optimize for space, use `CompactPtrHash`.
let mphf = <CompactPtrHash>::new(&keys, PtrHashParams::default_compact());
```

## Epserde

The `PtrHash` datastructure can be (de)serialized to/from disk using
[epserde](https://github.com/vigna/epserde-rs) when the `epserde` feature is set.
This also allows convenient deserialization using `mmap`.
See [examples/epserde.rs](examples/epserde.rs) for an example.

## Sharding

In order to build PtrHash on large sets of keys that do not fit in ram, the keys
can be sharded and constructed one shard at a time.
See `fn sharding()` in [examples/evals.rs](examples/epserde.rs) for an example.

## Compared to PTHash

PtrHash extends PTHash in a few ways:

-   **8-bit pilots:** Instead of allowing pilots to take any integer value, we
    restrict them to `[0, 256)` and store them as `Vec<u8>` directly.
    This avoids the need for a compact or dictionary encoding.
-   **Evicting:** To get all pilots to be small, we use *evictions*, similar
    to *cuckoo hashing*: Whenever we cannot find a collision-free pilot for a
    bucket, we find the pilot with the fewest collisions and *evict* all
    colliding buckets, which are pushed on a queue after which they will search
    for a new pilot.
-   **Partitioning:** To speed up construction, we partition all keys/hashes
    into parts such that each part contains `S=2^k` *slots*.
    This significantly speeds up
    construction since all reads of the `taken` bitvector are now very local.
    
    This brings the benefit that the only global memory needed is to store the
    hashes for each part. The sorting, bucketing, and slot filling is per-part
    and needs comparatively little memory.
-   **Remap encoding:** We use the `CachelineEF` partitioned Elias-Fano encoding that stores
    chunks of `44` integers into a single cacheline. This takes `~30%` more
    space for remapping, but replaces the three reads needed by (global)
    Elias-Fano encoding by a single read.
