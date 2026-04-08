#[macro_use]
extern crate criterion;

mod common;

use criterion::{Criterion, Throughput};
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::models::TrainerWrapper;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::tokenizer::{AddedToken, EncodeInput};
use tokenizers::Tokenizer;

use common::{iter_bench_encode, iter_bench_encode_batch, iter_bench_train};
use std::ops::Deref;

static BATCH_SIZE: usize = 1_000;

fn create_gpt2_tokenizer(bpe: BPE) -> Tokenizer {
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));
    tokenizer.with_decoder(Some(ByteLevel::default()));
    tokenizer.add_tokens(&[AddedToken::from("ing", false).single_word(false)]);
    tokenizer.add_special_tokens(&[AddedToken::from("[ENT]", true).single_word(true)]);
    tokenizer
}

fn bench_gpt2(c: &mut Criterion) {
    let bpe = BPE::from_file("data/gpt2-vocab.json", "data/gpt2-merges.txt")
        .build()
        .unwrap();
    let tokenizer = create_gpt2_tokenizer(bpe);
    let mut lines: Vec<EncodeInput> = vec![];
    let mut batches: Vec<Vec<EncodeInput>> = vec![vec![]];
    let data = std::fs::read_to_string("data/big.txt").unwrap();
    for line in data.lines() {
        let line: EncodeInput = line.into();
        lines.push(line.clone());
        if batches.last().unwrap().len() >= BATCH_SIZE {
            batches.push(vec![]);
        }
        batches.last_mut().unwrap().push(line);
    }

    let mut group = c.benchmark_group("bpe-encode");
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.bench_function("BPE GPT2 encode", |b| {
        b.iter_custom(|iters| iter_bench_encode(iters, tokenizer.deref(), &lines))
    });

    group.bench_function("BPE GPT2 encode batch", |b| {
        b.iter_custom(|iters| iter_bench_encode_batch(iters, tokenizer.deref(), &batches))
    });

    let bpe = BPE::from_file("data/gpt2-vocab.json", "data/gpt2-merges.txt")
        .cache_capacity(0)
        .build()
        .unwrap();
    let tokenizer = create_gpt2_tokenizer(bpe);

    group.bench_function("BPE GPT2 encode, no cache", |b| {
        b.iter_custom(|iters| iter_bench_encode(iters, &tokenizer, &lines))
    });

    group.bench_function("BPE GPT2 encode batch, no cache", |b| {
        b.iter_custom(|iters| iter_bench_encode_batch(iters, &tokenizer, &batches))
    });
}

fn bench_train_small(c: &mut Criterion) {
    let mut trainer: TrainerWrapper = BpeTrainerBuilder::default()
        .show_progress(false)
        .build()
        .into();
    let mut tokenizer = Tokenizer::new(BPE::default()).into_inner();
    let mut group = c.benchmark_group("bpe-train-small");
    let data = std::fs::read_to_string("data/small.txt").unwrap();
    group.throughput(Throughput::Bytes(data.len() as u64));
    tokenizer.with_pre_tokenizer(Some(Whitespace {}));
    group.bench_function("BPE Train vocabulary (small)", |b| {
        b.iter_custom(|iters| {
            iter_bench_train(
                iters,
                &mut tokenizer,
                &mut trainer,
                vec!["data/small.txt".to_string()],
            )
        })
    });
}

fn bench_train_big(c: &mut Criterion) {
    let mut trainer: TrainerWrapper = BpeTrainerBuilder::default()
        .show_progress(false)
        .build()
        .into();
    let mut tokenizer = Tokenizer::new(BPE::default()).into_inner();
    tokenizer.with_pre_tokenizer(Some(Whitespace {}));
    let mut group = c.benchmark_group("bpe-train-large");
    let data = std::fs::read_to_string("data/big.txt").unwrap();
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.bench_function("BPE Train vocabulary (big)", |b| {
        b.iter_custom(|iters| {
            iter_bench_train(
                iters,
                &mut tokenizer,
                &mut trainer,
                vec!["data/big.txt".to_string()],
            )
        })
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = bench_gpt2
}
criterion_group! {
    name = benches_train_small;
    config = Criterion::default().sample_size(10);
    targets = bench_train_small
}
criterion_group! {
    name = benches_train_big;
    config = Criterion::default().sample_size(10);
    targets = bench_train_big
}
criterion_main!(benches, benches_train_small, benches_train_big);
