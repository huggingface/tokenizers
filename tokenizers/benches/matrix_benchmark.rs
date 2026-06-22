//! Standardized ``(batch_size × input_length)`` encode + decode matrix.
//!
//! Mirrors the knobs used by fastokens' ``examples/ablation.sh`` and
//! wordchipper's batch bench: per-sample token length is controlled directly
//! (not document length), and both encode and decode are measured on the same
//! samples so throughput numbers are comparable side-by-side.
//!
//! Samples are built from ``data/big.txt``: the file is tokenized once, the
//! resulting ids are truncated/repeated to the target length (following
//! fastokens' ``_adjust_tokens``) and decoded back to text.
//!
//! Run with:
//!
//!   cargo bench --bench matrix_benchmark
//!
//! Local tuning of the sweep via env vars:
//!
//!   MATRIX_BATCH_SIZES=1,8,32   MATRIX_INPUT_LENGTHS=128,1024,8192  cargo bench --bench matrix_benchmark

#[macro_use]
extern crate criterion;

use criterion::{BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tokenizers::Tokenizer;

const DEFAULT_BATCH_SIZES: &[usize] = &[1, 8, 32, 128];
const DEFAULT_INPUT_LENGTHS: &[usize] = &[128, 512, 2048, 8192];

fn parse_env_list(name: &str, default: &[usize]) -> Vec<usize> {
    match std::env::var(name) {
        Ok(s) => s
            .split(',')
            .filter_map(|x| x.trim().parse::<usize>().ok())
            .collect(),
        Err(_) => default.to_vec(),
    }
}

/// Truncate or cyclically repeat ``ids`` to reach exactly ``target`` length.
/// Mirrors fastokens' ``_adjust_tokens`` helper.
fn adjust_tokens(ids: &[u32], target: usize) -> Vec<u32> {
    if ids.len() >= target {
        return ids[..target].to_vec();
    }
    let mut out = Vec::with_capacity(target);
    while out.len() < target {
        let remaining = target - out.len();
        out.extend_from_slice(&ids[..remaining.min(ids.len())]);
    }
    out
}

/// Build a fixed-length sample at the text level.
///
/// We tokenize ``source`` once, reshape the ids to ``input_length`` tokens via
/// ``adjust_tokens``, then decode back to text. Re-encoding the decoded text
/// may not produce exactly ``input_length`` ids for every tokenizer, but the
/// byte length is stable and representative of real inputs at that scale.
fn build_sample(tok: &Tokenizer, source: &str, input_length: usize) -> String {
    let encoded = tok.encode(source, false).unwrap();
    let adjusted = adjust_tokens(encoded.get_ids(), input_length);
    tok.decode(&adjusted, false).unwrap()
}

fn bench_matrix(c: &mut Criterion) {
    let tokenizer = Tokenizer::from_file("data/llama-3-tokenizer.json").unwrap();
    let corpus = std::fs::read_to_string("data/big.txt").unwrap();

    let batch_sizes = parse_env_list("MATRIX_BATCH_SIZES", DEFAULT_BATCH_SIZES);
    let input_lengths = parse_env_list("MATRIX_INPUT_LENGTHS", DEFAULT_INPUT_LENGTHS);

    // Pre-build one sample per input_length (reused across batch sizes).
    let samples: Vec<(usize, String)> = input_lengths
        .iter()
        .map(|&l| (l, build_sample(&tokenizer, &corpus, l)))
        .collect();

    // ------- ENCODE (batch, offsets tracked) -------
    let mut group = c.benchmark_group("matrix/encode-batch");
    for &bs in &batch_sizes {
        for (input_length, sample) in &samples {
            let batch: Vec<&str> = (0..bs).map(|_| sample.as_str()).collect();
            let total_bytes: u64 = batch.iter().map(|s| s.len() as u64).sum();
            group.throughput(Throughput::Bytes(total_bytes));
            group.bench_with_input(
                BenchmarkId::new(format!("len{input_length}"), bs),
                &(batch.clone()),
                |b, batch| {
                    b.iter(|| {
                        black_box(
                            tokenizer
                                .encode_batch(black_box(batch.clone()), false)
                                .unwrap(),
                        )
                    })
                },
            );
        }
    }
    group.finish();

    // ------- ENCODE (batch, offset-free fast path) -------
    let mut group = c.benchmark_group("matrix/encode-batch-fast");
    for &bs in &batch_sizes {
        for (input_length, sample) in &samples {
            let batch: Vec<&str> = (0..bs).map(|_| sample.as_str()).collect();
            let total_bytes: u64 = batch.iter().map(|s| s.len() as u64).sum();
            group.throughput(Throughput::Bytes(total_bytes));
            group.bench_with_input(
                BenchmarkId::new(format!("len{input_length}"), bs),
                &(batch.clone()),
                |b, batch| {
                    b.iter(|| {
                        black_box(
                            tokenizer
                                .encode_batch_fast(black_box(batch.clone()), false)
                                .unwrap(),
                        )
                    })
                },
            );
        }
    }
    group.finish();

    // ------- DECODE (batch) -------
    let mut group = c.benchmark_group("matrix/decode-batch");
    for &bs in &batch_sizes {
        for (input_length, sample) in &samples {
            let batch: Vec<&str> = (0..bs).map(|_| sample.as_str()).collect();
            let encs = tokenizer.encode_batch_fast(batch.clone(), false).unwrap();
            let ids_per_sample: Vec<Vec<u32>> =
                encs.iter().map(|e| e.get_ids().to_vec()).collect();
            let total_tokens: u64 = ids_per_sample.iter().map(|v| v.len() as u64).sum();
            group.throughput(Throughput::Elements(total_tokens));
            group.bench_with_input(
                BenchmarkId::new(format!("len{input_length}"), bs),
                &ids_per_sample,
                |b, ids_per_sample| {
                    b.iter(|| {
                        // decode_batch takes Vec<Vec<u32>> by value in this API; clone per iter
                        let slices: Vec<&[u32]> =
                            ids_per_sample.iter().map(|v| v.as_slice()).collect();
                        black_box(tokenizer.decode_batch(&slices, false).unwrap())
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group! {
    name = matrix;
    config = Criterion::default().sample_size(15);
    targets = bench_matrix
}
criterion_main!(matrix);
