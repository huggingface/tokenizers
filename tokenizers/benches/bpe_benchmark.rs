#[macro_use]
extern crate criterion;

mod common;

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use criterion::Criterion;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::models::TrainerWrapper;
use tokenizers::pre_tokenizers::byte_level::{gpt2_regex_optimized, ByteLevel};
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::tokenizer::{AddedToken, EncodeInput};
use tokenizers::{NormalizedString, SplitDelimiterBehavior, Tokenizer};

use common::{iter_bench_encode, iter_bench_encode_batch, iter_bench_train};
use lazy_static::lazy_static;
use onig::Regex;
use std::ops::Deref;
use std::time::Duration;

static BATCH_SIZE: usize = 1_000;

fn create_gpt2_tokenizer(bpe: BPE) -> Tokenizer {
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(ByteLevel::default());
    tokenizer.with_decoder(ByteLevel::default());
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
    for line in BufReader::new(File::open(Path::new("data/big.txt")).unwrap()).lines() {
        let line: EncodeInput = line.unwrap().into();
        lines.push(line.clone());
        if batches.last().unwrap().len() >= BATCH_SIZE {
            batches.push(vec![]);
        }
        batches.last_mut().unwrap().push(line);
    }

    c.bench_function("BPE GPT2 encode", |b| {
        b.iter_custom(|iters| iter_bench_encode(iters, tokenizer.deref(), &lines))
    });

    c.bench_function("BPE GPT2 encode batch", |b| {
        b.iter_custom(|iters| iter_bench_encode_batch(iters, tokenizer.deref(), &batches))
    });

    let bpe = BPE::from_file("data/gpt2-vocab.json", "data/gpt2-merges.txt")
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

fn bench_train(c: &mut Criterion) {
    let mut trainer: TrainerWrapper = BpeTrainerBuilder::default()
        .show_progress(false)
        .build()
        .into();
    let mut tokenizer = Tokenizer::new(BPE::default()).into_inner();
    tokenizer.with_pre_tokenizer(Whitespace::default());
    c.bench_function("BPE Train vocabulary (small)", |b| {
        b.iter_custom(|iters| {
            iter_bench_train(
                iters,
                &mut tokenizer,
                &mut trainer,
                vec!["data/small.txt".to_string()],
            )
        })
    });

    let mut tokenizer = Tokenizer::new(BPE::default()).into_inner();
    tokenizer.with_pre_tokenizer(Whitespace::default());
    c.bench_function("BPE Train vocabulary (big)", |b| {
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

lazy_static! {
    static ref RE: Regex =
        Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
            .unwrap();
}

fn bench_gpt_regex(c: &mut Criterion) {
    let re_ref: &Regex = &RE;
    let mut lines: Vec<NormalizedString> = vec![];
    for line in BufReader::new(File::open(Path::new("data/big.txt")).unwrap()).lines() {
        let input = line.unwrap();
        lines.push(NormalizedString::from(input.trim().clone()));
    }
    c.bench_function("BPE split REGEX", |b| {
        b.iter(|| {
            for normalized in &lines {
                normalized
                    .split(re_ref, SplitDelimiterBehavior::Isolated)
                    .unwrap();
            }
        })
    });
}

fn bench_gpt_logic(c: &mut Criterion) {
    let mut lines: Vec<NormalizedString> = vec![];
    for line in BufReader::new(File::open(Path::new("data/big.txt")).unwrap()).lines() {
        let input = line.unwrap();
        lines.push(NormalizedString::from(input.trim().clone()));
    }
    c.bench_function("BPE split Logic", |b| {
        b.iter(|| {
            for normalized in &lines {
                gpt2_regex_optimized(&normalized).unwrap();
            }
        })
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = bench_gpt2
}
criterion_group! {
    name = benches_train;
    config = Criterion::default().sample_size(10);
    targets = bench_train
}
criterion_group! {
    name = benches_regex;
    config = Criterion::default().sample_size(100).warm_up_time(Duration::from_secs(10));
    targets = bench_gpt_regex
}
criterion_group! {
    name = benches_regex_logic;
    config = Criterion::default().sample_size(100).warm_up_time(Duration::from_secs(10));
    targets = bench_gpt_logic
}
criterion_main!(benches, benches_train, benches_regex, benches_regex_logic);
