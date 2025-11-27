#[macro_use]
extern crate criterion;

mod common;

use common::{iter_bench_encode, iter_bench_encode_batch, iter_bench_train};
use criterion::{Criterion, Throughput};
use std::hint::black_box;
use tokenizers::{
    models::{bpe::BpeTrainerBuilder, TrainerWrapper},
    EncodeInput, Tokenizer,
};

static BATCH_SIZE: usize = 1_000;

pub fn llama3(c: &mut Criterion) {
    let data = std::fs::read_to_string("data/big.txt").unwrap();
    let mut group = c.benchmark_group("llama3-encode");
    group.throughput(Throughput::Bytes(data.len() as u64));
    let mut lines: Vec<EncodeInput> = vec![];
    let mut batches: Vec<Vec<EncodeInput>> = vec![vec![]];
    for line in data.lines() {
        let line: EncodeInput = line.into();
        lines.push(line.clone());
        if batches.last().unwrap().len() >= BATCH_SIZE {
            batches.push(vec![]);
        }
        batches.last_mut().unwrap().push(line);
    }
    let tokenizer = Tokenizer::from_file("data/llama-3-tokenizer.json").unwrap();
    group.bench_function("llama3-offsets", |b| {
        let data: Vec<_> = data.lines().collect();
        let add_special_tokens = false;
        b.iter(|| {
            tokenizer
                .encode_batch_char_offsets(black_box(data.clone()), add_special_tokens)
                .unwrap()
        })
    });
    group.bench_function("llama3-encode", |b| {
        b.iter_custom(|iters| iter_bench_encode(iters, &tokenizer, &lines))
    });
    group.bench_function("llama3-batch", |b| {
        b.iter_custom(|iters| iter_bench_encode_batch(iters, &tokenizer, &batches))
    });
    let mut trainer: TrainerWrapper = BpeTrainerBuilder::default()
        .show_progress(false)
        .build()
        .into();
    let mut tokenizer = Tokenizer::from_file("data/llama-3-tokenizer.json").unwrap();
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
    group.finish();
}

criterion_group! {
    name = llama_3;
    config = Criterion::default().sample_size(10);
    targets = llama3
}

criterion_main!(llama_3);
