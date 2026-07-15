use std::hint::black_box;

use super::*;
use crate::util::generate_keys;

/// Construct the MPHF and test all keys are mapped to unique indices.
#[test]
fn construct_random() {
    #[allow(unused_mut)]
    let mut ns = vec![
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 30, 100, 300, 1000, 3000, 10_000, 30_000, 100_000,
    ];
    #[cfg(not(debug_assertions))]
    ns.extend_from_slice(&[1_000_000, 3_000_000, 10_000_000, 20_000_000, 30_000_000]);

    for n in (0..10).chain(ns) {
        eprintln!("RANDOM Testing n = {}", n);
        let keys = generate_keys(n);
        let ptr_hash = DefaultPtrHash::<FastIntHash, _>::new(&keys, PtrHashParams::default_fast());
        let mut done = bitvec![0; n];
        for key in &keys {
            let idx = ptr_hash.index(&key);
            assert!(!done[idx]);
            done.set(idx, true);
        }
        let ptr_hash = FastPtrHash::<FastIntHash, _>::new(&keys, PtrHashParams::default_fast());
        let mut done = bitvec![0; ptr_hash.max_index()];
        for key in &keys {
            let idx = ptr_hash.index(key);
            assert!(!done[idx]);
            done.set(idx, true);
        }
        let ptr_hash =
            CompactPtrHash::<FastIntHash, _>::new(&keys, PtrHashParams::default_compact());
        let mut done = bitvec![0; n];
        for key in &keys {
            let idx = ptr_hash.index(key);
            assert!(!done[idx]);
            done.set(idx, true);
        }
    }
}

/// Construct the MPHF and test all keys are mapped to unique indices.
#[test]
#[ignore = "large"]
fn test_1e9() {
    env_logger::init();
    let n = 1_000_000_000;
    eprintln!("RANDOM Testing n = {}", n);
    let keys = generate_keys(n);

    let ptr_hash = DefaultPtrHash::<FastIntHash, _>::new(&keys, PtrHashParams::default_fast());
    let mut done = bitvec![0; n];
    for key in &keys {
        let idx = ptr_hash.index(key);
        assert!(!done[idx]);
        done.set(idx, true);
    }

    let ptr_hash = FastPtrHash::<FastIntHash, _>::new(&keys, PtrHashParams::default_fast());
    let mut done = bitvec![0; ptr_hash.max_index()];
    for key in &keys {
        let idx = ptr_hash.index(key);
        assert!(!done[idx]);
        done.set(idx, true);
    }

    let ptr_hash = CompactPtrHash::<FastIntHash, _>::new(&keys, PtrHashParams::default_compact());
    let mut done = bitvec![0; n];
    for key in &keys {
        let idx = ptr_hash.index(key);
        assert!(!done[idx]);
        done.set(idx, true);
    }
}

#[test]
#[ignore = "for benchmarking only"]
fn int_hash_speed() {
    let n = 10000000;
    let keys = (0..n as u64).map(|i| hash::C * i).collect::<Vec<_>>();
    let seed = black_box(132);

    let start = std::time::Instant::now();
    for k in &keys {
        black_box(FastIntHash::hash(k, seed));
    }
    eprintln!("Time {:?}", start.elapsed());

    let start = std::time::Instant::now();
    for k in &keys {
        black_box(StrongerIntHash::hash(k, seed));
    }
    eprintln!("Time {:?}", start.elapsed());

    #[cfg(feature = "gxhash")]
    {
        let start = std::time::Instant::now();
        for k in &keys {
            black_box(GxInt::hash(k, seed));
        }
        eprintln!("Time {:?}", start.elapsed());
    }

    let start = std::time::Instant::now();
    for k in &keys {
        black_box(Xxh3Int::hash(k, seed));
    }
    eprintln!("Time {:?}", start.elapsed());
}

/// Keys are multiples of 1, 2^40, and 10^12
#[test]
fn construct_multiples() {
    env_logger::init();
    for m in [1, 1 << 40, 1_000_000_000_000, 3u64.pow(23)] {
        #[allow(unused_mut)]
        let mut ns = vec![
            0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 30, 100, 300, 1000, 3000, 10_000, 30_000,
            100_000,
        ];
        #[cfg(not(debug_assertions))]
        ns.extend_from_slice(&[1_000_000, 3_000_000, 10_000_000, 20_000_000, 30_000_000]);

        for n in ns {
            eprintln!("MULTIPLES OF {m} Testing n = {}", n);
            let mut keys = (0..n as u64).map(|i| m * i).collect::<Vec<_>>();
            keys.sort();
            for i in 0..n.saturating_sub(1) {
                if keys[i] >= keys[i + 1] {
                    keys[i + 1] = keys[i] + 1;
                }
            }
            let ptr_hash =
                DefaultPtrHash::<StrongerIntHash, _>::new(&keys, PtrHashParams::default_fast());
            let mut done = bitvec![0; n];
            for key in keys {
                let idx = ptr_hash.index(&key);
                assert!(!done[idx]);
                done.set(idx, true);
            }
        }
    }
}

#[test]
fn index_stream() {
    for n in [2, 10, 100, 1000, 10_000, 100_000, 1_000_000] {
        let keys = generate_keys(n);
        let ptr_hash = <DefaultPtrHash>::new(&keys, Default::default());
        let sum = ptr_hash.index_stream::<32, _>(&keys).sum::<usize>();
        assert_eq!(sum, (n * (n - 1)) / 2, "Failure for n = {n}");
    }
}

#[cfg(feature = "unstable")]
#[test]
fn index_batch() {
    for n in [10usize, 100, 1000, 10_000, 100_000, 1_000_000] {
        let n = n.next_multiple_of(32);
        let keys = generate_keys(n);
        let ptr_hash = <PtrHash>::new(&keys, Default::default());
        let sum = ptr_hash.index_batch_exact::<32>(&keys).sum::<usize>();
        assert_eq!(sum, (n * (n - 1)) / 2);
    }
}

#[test]
fn new_par_iter() {
    let n = 100_000;
    let keys = generate_keys(n);
    <PtrHash>::new_from_par_iter(n, keys.par_iter(), Default::default());
}

#[test]
#[cfg_attr(debug_assertions, ignore = "only run in release mode")]
fn in_memory_sharding() {
    let n = 1 << 25;
    let range = 0..n as u64;
    let keys = range.clone().into_par_iter();
    let ptr_hash = <CompactPtrHash<StrongerIntHash>>::new_from_par_iter(
        n,
        keys.clone(),
        PtrHashParams {
            keys_per_shard: 1 << 22,
            sharding: Sharding::Memory,
            ..PtrHashParams::default_compact()
        },
    );
    eprintln!("Checking duplicates...");
    let mut done = bitvec![0; n];
    for key in range {
        let idx = ptr_hash.index(&key);
        assert!(!done[idx]);
        done.set(idx, true);
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore = "only run in release mode")]
fn on_disk_sharding() {
    let n = 1 << 25;
    let range = 0..n as u64;
    let keys = range.clone().into_par_iter();
    let ptr_hash = <CompactPtrHash<StrongerIntHash>>::new_from_par_iter(
        n,
        keys.clone(),
        PtrHashParams {
            keys_per_shard: 1 << 22,
            sharding: Sharding::Disk,
            ..PtrHashParams::default_compact()
        },
    );
    eprintln!("Checking duplicates...");
    let mut done = bitvec![0; n];
    for key in range {
        let idx = ptr_hash.index(&key);
        assert!(!done[idx]);
        done.set(idx, true);
    }
}

/// Test that sharded construction and queries work with more than 2^32 keys.
#[test]
#[ignore = "very slow"]
fn many_keys_memory() {
    let n = 1 << 33;
    let n_query = 1 << 27;
    let range = 0..n as u64;
    let keys = range.clone().into_par_iter();
    let ptr_hash = <CompactPtrHash<StrongerIntHash>>::new_from_par_iter(
        n,
        keys.clone(),
        PtrHashParams {
            keys_per_shard: 1 << 30,
            sharding: Sharding::Memory,
            ..PtrHashParams::default_compact()
        },
    );
    // Since running all queries is super slow, we only check a subset of them.
    // Although this doesn't completely check that there are no duplicate
    // mappings, by the birthday paradox we can be quite sure there are none
    // since we check way more than sqrt(n) of them.
    eprintln!("Checking duplicates...");
    let mut done = bitvec![0; n];
    for key in 0..n_query {
        let idx = ptr_hash.index(&key);
        assert!(!done[idx]);
        done.set(idx, true);
    }
}

/// Test that sharded construction and queries work with more than 2^32 keys.
#[test]
#[ignore = "very slow; writes 64GB to disk"]
fn many_keys_disk() {
    let n = 1 << 33;
    let n_query = 1 << 27;
    let range = 0..n as u64;
    let keys = range.clone().into_par_iter();
    let ptr_hash = <CompactPtrHash<StrongerIntHash>>::new_from_par_iter(
        n,
        keys.clone(),
        PtrHashParams {
            keys_per_shard: 1 << 30,
            sharding: Sharding::Disk,
            ..PtrHashParams::default_compact()
        },
    );
    // Since running all queries is super slow, we only check a subset of them.
    // Although this doesn't completely check that there are no duplicate
    // mappings, by the birthday paradox we can be quite sure there are none
    // since we check way more than sqrt(n) of them.
    eprintln!("Checking duplicates...");
    let mut done = bitvec![0; n];
    for key in 0..n_query {
        let idx = ptr_hash.index(&key);
        assert!(!done[idx]);
        done.set(idx, true);
    }
}

#[test]
fn ptr_hash_can_clone() {
    let ptr_hash = PtrHash::<_>::new(&[0, 1], PtrHashParams::default());

    // test succeeds if this compiles
    let _y = ptr_hash.clone();
}

#[test]
fn integer_key_types() {
    let h = PtrHash::<_>::new(&[0u8], PtrHashParams::default());
    h.index(&0u8);
    let h = PtrHash::<_>::new(&[0u16], PtrHashParams::default());
    h.index(&0u16);
    let h = PtrHash::<_>::new(&[0u32], PtrHashParams::default());
    h.index(&0u32);
    let h = PtrHash::<_>::new(&[0u64], PtrHashParams::default());
    h.index(&0u64);
    let h = PtrHash::<_>::new(&[0usize], PtrHashParams::default());
    h.index(&0usize);
    let h = PtrHash::<_>::new(&[0i8], PtrHashParams::default());
    h.index(&0i8);
    let h = PtrHash::<_>::new(&[0i16], PtrHashParams::default());
    h.index(&0i16);
    let h = PtrHash::<_>::new(&[0i32], PtrHashParams::default());
    h.index(&0i32);
    let h = PtrHash::<_>::new(&[0i64], PtrHashParams::default());
    h.index(&0i64);
    let h = PtrHash::<_>::new(&[0isize], PtrHashParams::default());
    h.index(&0isize);
}

#[test]
#[cfg(feature = "gxhash")]
fn string_key_types() {
    let h = DefaultPtrHash::<StringHash, &str>::new(&["a"], PtrHashParams::default());

    // h.index("a");
    h.index(&"a");
    h.index(&"a".to_string().as_str());
    h.index(&Box::new("a"));

    // TODO: The below don't work yet.
    // See https://github.com/beling/bsuccinct-rs/issues/9 for some comments.

    // let h = DefaultPtrHash::<StringHash, &str>::new(&["a".to_string()], PtrHashParams::default());

    // h.index(&&"a");
    // h.index(&"a");
    // h.index("a".to_string());
    // h.index(&"a".to_string());
    // h.index(Box::new("a"));
    // h.index(&Box::new("a"));
    // h.index(Box::new("a".to_string()));
    // h.index(&Box::new("a".to_string()));

    // let h = DefaultPtrHash::<StringHash, _>::new(&[Box::new("a")], PtrHashParams::default());
    // h.index("a");
    // h.index(&"a");
    // h.index("a".to_string());
    // h.index(&"a".to_string());
    // h.index(Box::new("a"));
    // h.index(&Box::new("a"));
    // h.index(Box::new("a".to_string()));
    // h.index(&Box::new("a".to_string()));

    // let h = DefaultPtrHash::<StringHash, _>::new(
    //     &[Box::new("a".to_string())],
    //     PtrHashParams::default(),
    // );
    // h.index("a");
    // h.index(&"a");
    // h.index("a".to_string());
    // h.index(&"a".to_string());
    // h.index(Box::new("a"));
    // h.index(&Box::new("a"));
    // h.index(Box::new("a".to_string()));
    // h.index(&Box::new("a".to_string()));
}

#[test]
fn single_part() {
    let n = 1_000_000;
    let keys = util::generate_keys(n);

    let params = PtrHashParams::default();

    let mphf = <PtrHash<u64, _, Vec<u32>, hash::FastIntHash, _, true, true>>::new(&keys, params);

    mphf.index(&0);
}
