#[macro_use]
extern crate criterion;

use criterion::{Criterion, Throughput};
use itertools::Itertools;
use tokenizers::models::backtracking_bpe;
use tokenizers::PreTokenizerWrapper;
use tokenizers::Tokenizer;

pub fn llama3(c: &mut Criterion) {
    let data = std::fs::read_to_string("data/big.txt").unwrap();
    let mut group = c.benchmark_group("llama3-encode");
    group.throughput(Throughput::Bytes(data.bytes().len() as u64));

    group.bench_function("llama3-backtracking", |b| {
        let mut tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();
        let data: Vec<_> = data.lines().collect();
        let add_special_tokens = false;
        b.iter(|| {
            tokenizer
                .encode_batch_fast(criterion::black_box(data.clone()), add_special_tokens)
                .unwrap()
        })
    });

    group.bench_function("llama3-backtracking-no-pretok", |b| {
        let mut tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();
        tokenizer.with_pre_tokenizer(None::<PreTokenizerWrapper>);
        let data: Vec<_> = data.lines().collect();
        let add_special_tokens = false;
        b.iter(|| {
            tokenizer
                .encode_batch_fast(criterion::black_box(data.clone()), add_special_tokens)
                .unwrap()
        })
    });

    group.bench_function("llama3-encode_batch_fast", |b| {
        let tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();
        let data: Vec<_> = data.lines().collect();
        let add_special_tokens = false;
        b.iter(|| {
            tokenizer
                .encode_batch_fast(criterion::black_box(data.clone()), add_special_tokens)
                .unwrap()
        })
    });
    group.bench_function("llama3-encode_batch", |b| {
        let tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();
        let data: Vec<_> = data.lines().collect();
        let add_special_tokens = false;
        b.iter(|| {
            tokenizer
                .encode_batch(criterion::black_box(data.clone()), add_special_tokens)
                .unwrap()
        })
    });

    group.finish();
}

criterion_group! {
    name = llama;
    config = Criterion::default().sample_size(10);
    targets = llama3
}

criterion_main!(llama);
