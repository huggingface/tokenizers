//! Consolidated benchmark suite for CI regression detection.
//! Results are stored on HF Hub at hf-internal-testing/tokenizers-bench.
//!
//! Covers the key performance surfaces in a single binary:
//!   - BPE GPT-2 encode (single, batch, cached, uncached)
//!   - Llama-3 encode (single, batch, fast, concurrent)
//!   - Serialization round-trip (load + save)
//!   - BPE training (small corpus)
//!
//! Run locally:
//!
//!   cargo bench --bench ci_benchmark
//!
//! In CI the workflow pipes `--output-format bencher` to produce machine-readable
//! output that `github-action-benchmark` can compare against a stored baseline.

#[macro_use]
extern crate criterion;

mod common;

use common::{iter_bench_encode, iter_bench_encode_batch, iter_bench_train};
use criterion::{Criterion, Throughput};
use std::hint::black_box;
use std::ops::Deref;
use std::sync::Arc;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::models::TrainerWrapper;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::tokenizer::{AddedToken, EncodeInput};
use tokenizers::Tokenizer;

static BATCH_SIZE: usize = 1_000;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read big.txt and split into owned lines and batches.
/// Returns `(raw_data, lines, batches)` where lines/batches own their strings.
fn load_data() -> (
    String,
    Vec<EncodeInput<'static>>,
    Vec<Vec<EncodeInput<'static>>>,
) {
    let data = std::fs::read_to_string("data/big.txt").unwrap();
    let mut lines: Vec<EncodeInput> = vec![];
    let mut batches: Vec<Vec<EncodeInput>> = vec![vec![]];
    // Convert each line to an owned String so the EncodeInput is 'static
    for line in data.lines() {
        let owned: EncodeInput = line.to_owned().into();
        lines.push(owned.clone());
        if batches.last().unwrap().len() >= BATCH_SIZE {
            batches.push(vec![]);
        }
        batches.last_mut().unwrap().push(owned);
    }
    (data, lines, batches)
}

fn create_gpt2_tokenizer(bpe: BPE) -> Tokenizer {
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));
    tokenizer.with_decoder(Some(ByteLevel::default()));
    tokenizer
        .add_tokens([AddedToken::from("ing", false).single_word(false)])
        .unwrap();
    tokenizer
        .add_special_tokens([AddedToken::from("[ENT]", true).single_word(true)])
        .unwrap();
    tokenizer
}

// ---------------------------------------------------------------------------
// BPE GPT-2
// ---------------------------------------------------------------------------

fn bench_bpe_gpt2(c: &mut Criterion) {
    let (data, lines, batches) = load_data();

    let bpe = BPE::from_file("data/gpt2-vocab.json", "data/gpt2-merges.txt")
        .build()
        .unwrap();
    let tokenizer = create_gpt2_tokenizer(bpe);

    let mut group = c.benchmark_group("bpe-gpt2");
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_function("encode", |b| {
        b.iter_custom(|iters| iter_bench_encode(iters, tokenizer.deref(), &lines))
    });
    group.bench_function("encode-batch", |b| {
        b.iter_custom(|iters| iter_bench_encode_batch(iters, tokenizer.deref(), &batches))
    });

    // No-cache variant: stresses the merge loop directly
    let bpe_nc = BPE::from_file("data/gpt2-vocab.json", "data/gpt2-merges.txt")
        .cache_capacity(0)
        .build()
        .unwrap();
    let tok_nc = create_gpt2_tokenizer(bpe_nc);
    group.bench_function("encode-no-cache", |b| {
        b.iter_custom(|iters| iter_bench_encode(iters, &tok_nc, &lines))
    });
    group.bench_function("encode-batch-no-cache", |b| {
        b.iter_custom(|iters| iter_bench_encode_batch(iters, &tok_nc, &batches))
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Llama-3
// ---------------------------------------------------------------------------

fn bench_llama3(c: &mut Criterion) {
    let (data, lines, batches) = load_data();
    let tokenizer = Tokenizer::from_file("data/llama-3-tokenizer.json").unwrap();
    let tokenizer_arc = Arc::new(tokenizer.clone());

    let mut group = c.benchmark_group("llama3");
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_function("encode", |b| {
        b.iter_custom(|iters| iter_bench_encode(iters, &tokenizer, &lines))
    });
    group.bench_function("encode-batch", |b| {
        b.iter_custom(|iters| iter_bench_encode_batch(iters, &tokenizer, &batches))
    });

    // encode_fast: skips offset tracking (OffsetType::None)
    group.bench_function("encode-fast", |b| {
        let data_lines: Vec<&str> = data.lines().collect();
        b.iter(|| {
            for line in &data_lines {
                black_box(tokenizer.encode_fast(black_box(*line), false).unwrap());
            }
        })
    });

    // encode_batch_char_offsets: stresses the offset-tracking path
    group.bench_function("encode-char-offsets", |b| {
        let data_lines: Vec<&str> = data.lines().collect();
        b.iter(|| {
            tokenizer
                .encode_batch_char_offsets(black_box(data_lines.clone()), false)
                .unwrap()
        })
    });

    // Concurrent: 4 threads, each encodes a long chunk (~80k chars)
    let all_lines: Vec<&str> = data.lines().collect();
    let lines_per_thread = 1000;
    let num_threads = 4;
    let inputs: Vec<String> = (0..num_threads)
        .map(|i| {
            let start = i * lines_per_thread;
            all_lines[start..start + lines_per_thread].join("\n")
        })
        .collect();
    let total_bytes: usize = inputs.iter().map(|s| s.len()).sum();
    let tok = tokenizer_arc.clone();
    group.throughput(Throughput::Bytes(total_bytes as u64));
    group.bench_function("concurrent-4t", move |b| {
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

    group.finish();
}

// ---------------------------------------------------------------------------
// Serialization round-trip
// ---------------------------------------------------------------------------

fn bench_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");

    // Tokenizer::from_file — the real user-facing load path
    group.bench_function("from-file-roberta", |b| {
        b.iter(|| black_box(Tokenizer::from_file("data/roberta.json").unwrap()))
    });
    group.bench_function("from-file-llama3", |b| {
        b.iter(|| black_box(Tokenizer::from_file("data/llama-3-tokenizer.json").unwrap()))
    });
    group.bench_function("from-file-albert", |b| {
        b.iter(|| black_box(Tokenizer::from_file("data/albert-base-v1-tokenizer.json").unwrap()))
    });

    // BPE::from_file — raw vocab+merges loading (no tokenizer wrapper)
    group.bench_function("bpe-from-file-gpt2", |b| {
        b.iter(|| {
            black_box(
                BPE::from_file("data/gpt2-vocab.json", "data/gpt2-merges.txt")
                    .build()
                    .unwrap(),
            )
        })
    });

    // Save path
    let llama3_tok = Tokenizer::from_file("data/llama-3-tokenizer.json").unwrap();
    let llama3_json = serde_json::to_string(&llama3_tok).unwrap();
    group.throughput(Throughput::Bytes(llama3_json.len() as u64));
    group.bench_function("save-llama3", |b| {
        b.iter(|| {
            let json = serde_json::to_string(black_box(&llama3_tok)).unwrap();
            black_box(json);
        })
    });

    // Deserialize from JSON string (no file I/O)
    let roberta_json = std::fs::read_to_string("data/roberta.json").unwrap();
    group.bench_function("deserialize-roberta", |b| {
        b.iter(|| black_box(serde_json::from_str::<Tokenizer>(&roberta_json).unwrap()))
    });
    group.bench_function("deserialize-llama3", |b| {
        b.iter(|| black_box(serde_json::from_str::<Tokenizer>(&llama3_json).unwrap()))
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Training (small corpus — just to catch regressions, not thorough)
// ---------------------------------------------------------------------------

fn bench_train(c: &mut Criterion) {
    let data = std::fs::read_to_string("data/small.txt").unwrap();
    let mut group = c.benchmark_group("train");
    group.throughput(Throughput::Bytes(data.len() as u64));

    let mut trainer: TrainerWrapper = BpeTrainerBuilder::default()
        .show_progress(false)
        .build()
        .into();
    let mut tokenizer = Tokenizer::new(BPE::default()).into_inner();
    tokenizer.with_pre_tokenizer(Some(Whitespace {}));

    group.bench_function("bpe-small", |b| {
        b.iter_custom(|iters| {
            iter_bench_train(
                iters,
                &mut tokenizer,
                &mut trainer,
                vec!["data/small.txt".to_string()],
            )
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group! {
    name = ci_bpe;
    config = Criterion::default().sample_size(15);
    targets = bench_bpe_gpt2
}
criterion_group! {
    name = ci_llama3;
    config = Criterion::default().sample_size(15);
    targets = bench_llama3
}
criterion_group! {
    name = ci_serial;
    config = Criterion::default().sample_size(15);
    targets = bench_serialization
}
criterion_group! {
    name = ci_train;
    config = Criterion::default().sample_size(15);
    targets = bench_train
}

criterion_main!(ci_bpe, ci_llama3, ci_serial, ci_train);
