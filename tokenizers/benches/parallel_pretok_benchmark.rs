#[macro_use]
extern crate criterion;

use criterion::{BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tokenizers::pattern::Pattern;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::utils::SysRegex;
use tokenizers::{PreTokenizedString, PreTokenizer};

/// GPT-2 byte-level regex pattern — the most common pre-tokenization regex
fn gpt2_regex() -> SysRegex {
    SysRegex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
        .unwrap()
}

fn bench_parallel_pretok(c: &mut Criterion) {
    let data = std::fs::read_to_string("data/big.txt").unwrap();

    // --- Raw find_matches: sequential vs parallel ---
    {
        let mut group = c.benchmark_group("find_matches");
        group.throughput(Throughput::Bytes(data.len() as u64));

        let re = gpt2_regex();

        group.bench_function("sequential", |b| {
            tokenizers::parallelism::set_parallelism(false);
            b.iter(|| (&re).find_matches(black_box(&data)).unwrap())
        });

        group.bench_function("parallel", |b| {
            tokenizers::parallelism::set_parallelism(true);
            b.iter(|| (&re).find_matches(black_box(&data)).unwrap())
        });

        // Restore default
        tokenizers::parallelism::set_parallelism(true);
        group.finish();
    }

    // --- Full pre-tokenizer pipeline: sequential vs parallel ---
    {
        let mut group = c.benchmark_group("byte-level-pretok");
        group.throughput(Throughput::Bytes(data.len() as u64));

        let pretok = ByteLevel::default();

        group.bench_function("sequential", |b| {
            tokenizers::parallelism::set_parallelism(false);
            b.iter(|| {
                let mut pre = PreTokenizedString::from(black_box(data.as_str()));
                pretok.pre_tokenize(&mut pre).unwrap();
                pre
            })
        });

        group.bench_function("parallel", |b| {
            tokenizers::parallelism::set_parallelism(true);
            b.iter(|| {
                let mut pre = PreTokenizedString::from(black_box(data.as_str()));
                pretok.pre_tokenize(&mut pre).unwrap();
                pre
            })
        });

        tokenizers::parallelism::set_parallelism(true);
        group.finish();
    }

    // --- Scaling by input size ---
    {
        let mut group = c.benchmark_group("parallel-pretok-scaling");
        let re = gpt2_regex();

        for size in [1_000, 10_000, 100_000, 500_000] {
            let input: String = data.chars().take(size).collect();
            group.throughput(Throughput::Bytes(input.len() as u64));

            group.bench_with_input(BenchmarkId::new("sequential", size), &input, |b, input| {
                tokenizers::parallelism::set_parallelism(false);
                b.iter(|| (&re).find_matches(black_box(input.as_str())).unwrap())
            });

            group.bench_with_input(BenchmarkId::new("parallel", size), &input, |b, input| {
                tokenizers::parallelism::set_parallelism(true);
                b.iter(|| (&re).find_matches(black_box(input.as_str())).unwrap())
            });
        }

        tokenizers::parallelism::set_parallelism(true);
        group.finish();
    }
}

criterion_group! {
    name = parallel_pretok;
    config = Criterion::default().sample_size(20);
    targets = bench_parallel_pretok
}

criterion_main!(parallel_pretok);
