#[macro_use]
extern crate criterion;

use criterion::{BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tokenizers::pre_tokenizers::whitespace::{ManualWhitespaceSplit, Whitespace};
use tokenizers::{PreTokenizedString, PreTokenizer};

fn bench_pretokenizer(c: &mut Criterion) {
    let data = std::fs::read_to_string("data/big.txt").unwrap();
    let lines: Vec<&str> = data.lines().collect();

    let mut group = c.benchmark_group("whitespace-pretok");
    group.throughput(Throughput::Bytes(data.len() as u64));

    // Full corpus as a single string
    group.bench_function("regex/full-corpus", |b| {
        let pretok = Whitespace {};
        b.iter(|| {
            let mut pre = PreTokenizedString::from(black_box(data.as_str()));
            pretok.pre_tokenize(&mut pre).unwrap();
            pre
        })
    });

    group.bench_function("manual/full-corpus", |b| {
        let pretok = ManualWhitespaceSplit {};
        b.iter(|| {
            let mut pre = PreTokenizedString::from(black_box(data.as_str()));
            pretok.pre_tokenize(&mut pre).unwrap();
            pre
        })
    });

    // Line-by-line (many short strings — tests per-call overhead)
    group.bench_function("regex/line-by-line", |b| {
        let pretok = Whitespace {};
        b.iter(|| {
            for line in &lines {
                let mut pre = PreTokenizedString::from(black_box(*line));
                pretok.pre_tokenize(&mut pre).unwrap();
                black_box(&pre);
            }
        })
    });

    group.bench_function("manual/line-by-line", |b| {
        let pretok = ManualWhitespaceSplit {};
        b.iter(|| {
            for line in &lines {
                let mut pre = PreTokenizedString::from(black_box(*line));
                pretok.pre_tokenize(&mut pre).unwrap();
                black_box(&pre);
            }
        })
    });

    group.finish();

    // --- Scaling with input size ---

    let mut group = c.benchmark_group("whitespace-pretok-scaling");

    for size in [100, 1_000, 10_000, 100_000] {
        let input: String = data.chars().take(size).collect();
        group.throughput(Throughput::Bytes(input.len() as u64));

        group.bench_with_input(BenchmarkId::new("regex", size), &input, |b, input| {
            let pretok = Whitespace {};
            b.iter(|| {
                let mut pre = PreTokenizedString::from(black_box(input.as_str()));
                pretok.pre_tokenize(&mut pre).unwrap();
                pre
            })
        });

        group.bench_with_input(BenchmarkId::new("manual", size), &input, |b, input| {
            let pretok = ManualWhitespaceSplit {};
            b.iter(|| {
                let mut pre = PreTokenizedString::from(black_box(input.as_str()));
                pretok.pre_tokenize(&mut pre).unwrap();
                pre
            })
        });
    }

    group.finish();
}

criterion_group! {
    name = whitespace_pretok;
    config = Criterion::default().sample_size(50);
    targets = bench_pretokenizer
}

criterion_main!(whitespace_pretok);
