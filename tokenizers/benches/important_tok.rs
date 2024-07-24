use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tokenizers::tokenizer::Tokenizer;
extern crate criterion;

mod common;

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use tokenizers::tokenizer::EncodeInput;

use common::{iter_bench_encode, iter_bench_encode_batch};
use std::ops::Deref;
use std::time::{Duration, Instant};

static BATCH_SIZE: usize = usize::MAX;

fn bench_inference(c: &mut Criterion) {
    let tokenizer = Tokenizer::from_pretrained("mistralai/Mistral-7B-v0.1", None).unwrap();
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

    c.bench_function("mistral encode long input", |b| {
        b.iter_custom(|iters| iter_bench_encode(iters, tokenizer.deref(), &lines))
    });

    c.bench_function("encode single batch of very long input", |b| {
        b.iter_custom(|iters| iter_bench_encode_batch(iters, tokenizer.deref(), &batches))
    });

    c.bench_function("decode long input", |b| {
        b.iter_custom(|iters| {
            let mut duration = Duration::new(0, 0);
            let mut line_index: usize = 0;
            for _i in 0..iters {
                if line_index >= lines.len() {
                    line_index = 0;
                }
                let input = batches[0].clone();
                let start = Instant::now();
                let _ = black_box(tokenizer.encode_batch(input, false));
                duration = duration.checked_add(start.elapsed()).unwrap();
            }
            duration
        })
    });
}

criterion_group!(benches, bench_inference);
criterion_main!(benches);
