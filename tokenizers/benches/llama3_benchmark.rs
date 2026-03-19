#[macro_use]
extern crate criterion;

mod common;

use common::{iter_bench_encode, iter_bench_encode_batch, iter_bench_train};
use criterion::{Criterion, Throughput};
use std::hint::black_box;
use std::sync::Arc;
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

    // Cache effectiveness: encode the same medium-length input 1000 times
    group.bench_function("llama3-cache-repeated", |b| {
        let sample = data.lines().find(|l| l.len() > 200).unwrap_or("hello world");
        b.iter(|| {
            for _ in 0..1000 {
                let _ = black_box(tokenizer.encode(black_box(sample), false));
            }
        })
    });

    // Concurrent: N threads each encode a medium prompt
    let tokenizer_arc = Arc::new(tokenizer.clone());
    for num_threads in [2, 4, 8] {
        let tok = tokenizer_arc.clone();
        let sample: String = data.lines().take(3).collect::<Vec<_>>().join("\n");
        group.bench_function(format!("llama3-concurrent-{num_threads}t"), move |b| {
            b.iter(|| {
                std::thread::scope(|s| {
                    let handles: Vec<_> = (0..num_threads)
                        .map(|_| {
                            let tok = &tok;
                            let sample = &sample;
                            s.spawn(move || {
                                let _ = black_box(tok.encode(black_box(sample.as_str()), false));
                            })
                        })
                        .collect();
                    for h in handles {
                        h.join().unwrap();
                    }
                });
            })
        });
    }

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
