#[macro_use]
extern crate criterion;

mod common;

use common::{iter_bench_encode, iter_bench_encode_batch};
use criterion::{Criterion, Throughput};
use std::hint::black_box;
use std::sync::Arc;
use tokenizers::{EncodeInput, Tokenizer};

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

    // Concurrent long-context: N threads each encode a different large input (80k chars)
    // through a shared tokenizer. Each thread gets 1000 unique lines, simulating
    // concurrent inference requests. Stresses DFA cache contention in the regex
    // engine — per-thread regex copies avoid thrashing here.
    let all_lines: Vec<&str> = data.lines().collect();
    let lines_per_thread = 1000;
    let tokenizer_arc = Arc::new(tokenizer.clone());
    for num_threads in [1, 2, 4, 8] {
        // Pre-build per-thread inputs (each thread gets a different 1000-line chunk)
        let inputs: Vec<String> = (0..num_threads)
            .map(|i| {
                let start = i * lines_per_thread;
                all_lines[start..start + lines_per_thread].join("\n")
            })
            .collect();
        let total_bytes: usize = inputs.iter().map(|s| s.len()).sum();
        let tok = tokenizer_arc.clone();
        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_function(format!("llama3-concurrent-long-{num_threads}t"), move |b| {
            b.iter(|| {
                std::thread::scope(|s| {
                    let handles: Vec<_> = inputs
                        .iter()
                        .map(|input| {
                            let tok = &tok;
                            s.spawn(move || {
                                black_box(tok.encode(black_box(input.as_str()), false).unwrap())
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
    group.finish();
}

criterion_group! {
    name = llama_3;
    config = Criterion::default().sample_size(10);
    targets = llama3
}

criterion_main!(llama_3);
