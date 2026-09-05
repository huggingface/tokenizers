#[macro_use]
extern crate criterion;

use std::hint::black_box;
use std::time::{Duration, Instant};

use criterion::{Criterion, Throughput};
use tokenizers::normalizers::byte_level::ByteLevel;
use tokenizers::{NormalizedString, Normalizer};

/// Times the `ByteLevel` *normalizer*'s `normalize` over `big.txt`. Inputs are built outside
/// the timed region so we measure the transform itself, not input allocation. (No end-to-end
/// bench covers this path — every other bench wires `ByteLevel` as a pre-tokenizer.)
fn bench_bytelevel_normalizer(c: &mut Criterion) {
    let norm = ByteLevel::new();
    let mut group = c.benchmark_group("bytelevel-normalizer");
    for path in ["data/small.txt", "data/big.txt"] {
        let data = std::fs::read_to_string(path).unwrap();
        let lines: Vec<&str> = data.lines().collect();
        group.throughput(Throughput::Bytes(data.len() as u64));
        group.bench_function(format!("ByteLevel normalize {path}"), |b| {
            b.iter_custom(|iters| {
                let mut duration = Duration::new(0, 0);
                for _ in 0..iters {
                    // Pre-build the inputs so only `normalize` is timed.
                    let mut inputs: Vec<NormalizedString> =
                        lines.iter().map(|l| NormalizedString::from(*l)).collect();
                    let start = Instant::now();
                    for ns in inputs.iter_mut() {
                        norm.normalize(black_box(ns)).unwrap();
                    }
                    duration = duration.checked_add(start.elapsed()).unwrap();
                    black_box(&inputs);
                }
                duration
            })
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = bench_bytelevel_normalizer
}
criterion_main!(benches);
