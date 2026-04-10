#[macro_use]
extern crate criterion;

use criterion::{BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::tokenizer::{
    AddedToken, TruncationDirection, TruncationParams, TruncationStrategy,
};
use tokenizers::Tokenizer;

fn create_gpt2_tokenizer() -> Tokenizer {
    let bpe = BPE::from_file("data/gpt2-vocab.json", "data/gpt2-merges.txt")
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));
    tokenizer.with_decoder(Some(ByteLevel::default()));
    tokenizer
        .add_tokens([AddedToken::from("ing", false).single_word(false)])
        .unwrap();
    tokenizer
        .add_special_tokens(&[AddedToken::from("[ENT]", true).single_word(true)])
        .unwrap();
    tokenizer
}

fn bench_truncation(c: &mut Criterion) {
    let data = std::fs::read_to_string("data/big.txt").unwrap();

    let mut group = c.benchmark_group("truncation-waste");
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_function("no-truncation", |b| {
        let tokenizer = create_gpt2_tokenizer();
        b.iter(|| tokenizer.encode(black_box(data.as_str()), false).unwrap())
    });

    group.bench_function("truncate-to-512", |b| {
        let mut tokenizer = create_gpt2_tokenizer();
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: 512,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
                direction: TruncationDirection::Right,
            }))
            .unwrap();
        b.iter(|| tokenizer.encode(black_box(data.as_str()), false).unwrap())
    });

    let short_input: String = data.chars().take(2048).collect();
    group.bench_function("pre-truncated-input-512", |b| {
        let mut tokenizer = create_gpt2_tokenizer();
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: 512,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
                direction: TruncationDirection::Right,
            }))
            .unwrap();
        b.iter(|| {
            tokenizer
                .encode(black_box(short_input.as_str()), false)
                .unwrap()
        })
    });

    group.finish();

    let mut group = c.benchmark_group("truncation-scaling-by-input");

    let max_length = 512;
    for input_chars in [1_000, 10_000, 100_000, 500_000] {
        let input: String = data.chars().take(input_chars).collect();
        group.throughput(Throughput::Bytes(input.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("encode-then-truncate", input_chars),
            &input,
            |b, input| {
                let mut tokenizer = create_gpt2_tokenizer();
                tokenizer
                    .with_truncation(Some(TruncationParams {
                        max_length,
                        strategy: TruncationStrategy::LongestFirst,
                        stride: 0,
                        direction: TruncationDirection::Right,
                    }))
                    .unwrap();
                b.iter(|| tokenizer.encode(black_box(input.as_str()), false).unwrap())
            },
        );
    }

    group.finish();

    let mut group = c.benchmark_group("truncation-scaling-by-maxlen");
    let fixed_input: String = data.chars().take(50_000).collect();
    group.throughput(Throughput::Bytes(fixed_input.len() as u64));

    for max_len in [128, 512, 2048, 8192] {
        group.bench_with_input(
            BenchmarkId::new("max-length", max_len),
            &max_len,
            |b, &max_len| {
                let mut tokenizer = create_gpt2_tokenizer();
                tokenizer
                    .with_truncation(Some(TruncationParams {
                        max_length: max_len,
                        strategy: TruncationStrategy::LongestFirst,
                        stride: 0,
                        direction: TruncationDirection::Right,
                    }))
                    .unwrap();
                b.iter(|| {
                    tokenizer
                        .encode(black_box(fixed_input.as_str()), false)
                        .unwrap()
                })
            },
        );
    }

    group.finish();

    let mut group = c.benchmark_group("truncation-direction");
    let mid_input: String = data.chars().take(10_000).collect();
    group.throughput(Throughput::Bytes(mid_input.len() as u64));

    for (name, direction) in [
        ("right", TruncationDirection::Right),
        ("left", TruncationDirection::Left),
    ] {
        group.bench_function(name, |b| {
            let mut tokenizer = create_gpt2_tokenizer();
            tokenizer
                .with_truncation(Some(TruncationParams {
                    max_length: 512,
                    strategy: TruncationStrategy::LongestFirst,
                    stride: 0,
                    direction,
                }))
                .unwrap();
            b.iter(|| {
                tokenizer
                    .encode(black_box(mid_input.as_str()), false)
                    .unwrap()
            })
        });
    }

    group.finish();
}

criterion_group! {
    name = truncation_benches;
    config = Criterion::default().sample_size(20);
    targets = bench_truncation
}

criterion_main!(truncation_benches);
