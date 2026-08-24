//! Decode throughput: `decode`, `decode_batch` and `decode_stream`.
//!
//! Deliberately small. The cross-engine comparisons, the model matrix and the per-language
//! sweeps live in huggingface/tokbench; what stays here is enough to catch a regression in the
//! three decode entry points on one byte-level BPE, over Latin and over Japanese.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use tk_encode::pipeline::PipelineTokenizer;

const BATCH_SIZE: usize = 1_000;
// How many tokens to run through streaming decode pipeline
const STREAM_TOKEN_BUDGET: usize = 300_000;

/// Read a real config from `../data`. They are all still version `1.0`, so run the upgrade pass
/// first -- the reader only accepts canonical `2.0`.
fn load(path: &str) -> PipelineTokenizer {
    let canonical = tk_convert::canonicalize_file(path).unwrap();
    tk_serialize::from_json(&canonical).unwrap()
}

fn take_token_budget(token_sequences: &[Vec<u32>], budget: usize) -> &[Vec<u32>] {
    let mut total = 0;
    let end = token_sequences
        .iter()
        .position(|seq| {
            total += seq.len();
            total > budget
        })
        .unwrap_or(token_sequences.len());
    &token_sequences[..end]
}

/// Encode `data` line by line, so the decode benches have real id sequences to chew on.
fn encode_lines(tokenizer: &PipelineTokenizer, data: &str) -> Vec<Vec<u32>> {
    let lines: Vec<&str> = data.lines().collect();
    tokenizer
        .encode(lines.as_slice(), false)
        .wait()
        .unwrap()
        .iter()
        .map(|encoding| encoding.ids().iter().copied().map(u32::from).collect())
        .collect()
}

pub fn decode(c: &mut Criterion) {
    let big_txt = std::fs::read_to_string("../data/big.txt").unwrap();
    let japanese_txt = std::fs::read_to_string("../data/unigram_wagahaiwa_nekodearu.txt").unwrap();
    let tokenizer = load("../data/llama-3-tokenizer.json");

    for (name, data) in [
        ("decode-llama3-en", &big_txt),
        ("decode-llama3-ja", &japanese_txt),
    ] {
        // Build input: tokenize data and group it in lines and batches
        let lines = encode_lines(&tokenizer, data);
        // Long lines, for decode_stream
        let fused_lines: Vec<Vec<u32>> = lines
            .chunks(BATCH_SIZE)
            .map(|chunk| chunk.concat())
            .collect();
        let batches: Vec<Vec<&[u32]>> = lines
            .chunks(BATCH_SIZE)
            .map(|chunk| chunk.iter().map(Vec::as_slice).collect())
            .collect();

        let total_decoded_bytes: usize = lines
            .iter()
            .map(|line| tokenizer.decode(line, false).unwrap().len())
            .sum();

        let mut group = c.benchmark_group(name);
        group.sampling_mode(criterion::SamplingMode::Flat);
        group.sample_size(20);

        group.throughput(Throughput::Bytes(total_decoded_bytes as u64));
        group.bench_function("decode", |bencher| {
            bencher.iter(|| {
                for line in &lines {
                    black_box(tokenizer.decode(line, false).unwrap());
                }
            })
        });
        group.bench_function("decode_batch", |bencher| {
            bencher.iter(|| {
                for batch in &batches {
                    black_box(tokenizer.decode_batch(batch, false).unwrap());
                }
            })
        });

        let stream_lines = take_token_budget(&lines, STREAM_TOKEN_BUDGET);
        let stream_fused = take_token_budget(&fused_lines, STREAM_TOKEN_BUDGET);

        // Throughput expressed in tokens / sec
        group.throughput(Throughput::Elements(
            stream_lines.iter().map(|line| line.len()).sum::<usize>() as u64,
        ));
        group.bench_function("decode_stream, short lines", |bencher| {
            bencher.iter(|| {
                for line in stream_lines {
                    let mut stream = tokenizer.decode_stream(false);
                    for &id in line {
                        black_box(stream.step(id).unwrap());
                    }
                }
            })
        });
        // Throughput expressed in tokens / sec
        group.throughput(Throughput::Elements(
            stream_fused.iter().map(|line| line.len()).sum::<usize>() as u64,
        ));
        group.bench_function("decode_stream, long lines", |bencher| {
            bencher.iter(|| {
                for line in stream_fused {
                    let mut stream = tokenizer.decode_stream(false);
                    for &id in line {
                        black_box(stream.step(id).unwrap());
                    }
                }
            })
        });
    }
}

criterion_group!(decode_benches, decode);
criterion_main!(decode_benches);
