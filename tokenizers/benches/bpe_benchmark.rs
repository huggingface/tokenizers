#[macro_use]
extern crate criterion;

use criterion::{black_box, Criterion};
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use std::time::{Duration, Instant};
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::tokenizer::{AddedToken, EncodeInput, Tokenizer};

static BATCH_SIZE: usize = 1_000;

fn create_gpt2_tokenizer(bpe: BPE) -> Tokenizer {
    let mut tokenizer = Tokenizer::new(Box::new(bpe));
    tokenizer.with_pre_tokenizer(Box::new(ByteLevel::new(true)));
    tokenizer.with_decoder(Box::new(ByteLevel::new(false)));
    tokenizer.add_tokens(&[
        AddedToken {
            content: String::from("ing"),
            single_word: false,
        },
        AddedToken {
            content: String::from("[ENT]"),
            single_word: true,
        },
    ]);
    tokenizer
}

fn line_to_input(line: io::Result<String>) -> EncodeInput {
    EncodeInput::Single(line.unwrap())
}

fn iter_bench_encode(iters: u64, tokenizer: &Tokenizer, lines: &[EncodeInput]) -> Duration {
    let mut duration = Duration::new(0, 0);
    let mut line_index: usize = 0;
    for _i in 0..iters {
        if line_index >= lines.len() {
            line_index = 0;
        }
        let input = lines[line_index].clone();
        let start = Instant::now();
        let _ = black_box(tokenizer.encode(input));
        duration = duration.checked_add(start.elapsed()).unwrap();
    }
    duration
}

fn iter_bench_encode_batch(
    iters: u64,
    tokenizer: &Tokenizer,
    batches: &[Vec<EncodeInput>],
) -> Duration {
    let mut duration = Duration::new(0, 0);
    let mut batch_index: usize = 0;
    for _i in 0..iters {
        if batch_index >= batches.len() {
            batch_index = 0;
        }
        let batch = batches[batch_index].clone();
        let start = Instant::now();
        let _ = black_box(tokenizer.encode_batch(batch));
        duration = duration.checked_add(start.elapsed()).unwrap();
    }
    duration
}

fn bench_gpt2(c: &mut Criterion) {
    let bpe = BPE::from_files("benches/gpt2-vocab.json", "benches/gpt2-merges.txt")
        .unwrap()
        .build()
        .unwrap();
    let tokenizer = create_gpt2_tokenizer(bpe);
    let mut lines: Vec<EncodeInput> = vec![];
    let mut batches: Vec<Vec<EncodeInput>> = vec![vec![]];
    for line in BufReader::new(File::open(Path::new("benches/big.txt")).unwrap())
        .lines()
        .map(line_to_input)
    {
        lines.push(line.clone());
        if batches.last().unwrap().len() >= BATCH_SIZE {
            batches.push(vec![]);
        }
        batches.last_mut().unwrap().push(line);
    }

    c.bench_function("BPE GPT2 encode", |b| {
        b.iter_custom(|iters| iter_bench_encode(iters, &tokenizer, &lines))
    });

    c.bench_function("BPE GPT2 encode batch", |b| {
        b.iter_custom(|iters| iter_bench_encode_batch(iters, &tokenizer, &batches))
    });

    let bpe = BPE::from_files("benches/gpt2-vocab.json", "benches/gpt2-merges.txt")
        .unwrap()
        .cache_capacity(0)
        .build()
        .unwrap();
    let tokenizer = create_gpt2_tokenizer(bpe);

    c.bench_function("BPE GPT2 encode, no cache", |b| {
        b.iter_custom(|iters| iter_bench_encode(iters, &tokenizer, &lines))
    });

    c.bench_function("BPE GPT2 encode batch, no cache", |b| {
        b.iter_custom(|iters| iter_bench_encode_batch(iters, &tokenizer, &batches))
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = bench_gpt2
}
criterion_main!(benches);
