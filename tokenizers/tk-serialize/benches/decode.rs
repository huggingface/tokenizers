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

/// Text per `decode` call for the chunked bench, and the reason it exists.
///
/// The line-by-line benches are dominated by *per-call* cost: one allocation and one full
/// `from_utf8` validation for every short line. That hides anything happening inside the token
/// loop — measured, a change that cut the loop from 420 to 348 instructions moved them not at all.
/// 10 kB is what huggingface/tokbench feeds every engine, so a number measured here is both
/// sensitive to the loop and comparable with that matrix.
const CHUNK_BYTES: usize = 10 * 1024;

/// Encode `data` in ~`chunk_bytes` slices, split on char boundaries so no engine is ever handed
/// invalid UTF-8. Same chunking as tokbench's `core::chunk`.
fn encode_chunks(tokenizer: &PipelineTokenizer, data: &str, chunk_bytes: usize) -> Vec<Vec<u32>> {
    let mut chunks: Vec<&str> = Vec::new();
    let mut start = 0;
    while start < data.len() {
        let mut end = (start + chunk_bytes).min(data.len());
        while end < data.len() && !data.is_char_boundary(end) {
            end += 1;
        }
        chunks.push(&data[start..end]);
        start = end;
    }
    tokenizer
        .encode(chunks.as_slice(), false)
        .wait()
        .unwrap()
        .iter()
        .map(|encoding| encoding.ids().iter().copied().map(u32::from).collect())
        .collect()
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
        // The loop-sensitive one. See `CHUNK_BYTES`.
        let chunked = encode_chunks(&tokenizer, data, CHUNK_BYTES);
        let chunked_decoded_bytes: usize = chunked
            .iter()
            .map(|ids| tokenizer.decode(ids, false).unwrap().len())
            .sum();
        group.throughput(Throughput::Bytes(chunked_decoded_bytes as u64));
        group.bench_function("decode, 10 kB chunks", |bencher| {
            bencher.iter(|| {
                for ids in &chunked {
                    black_box(tokenizer.decode(ids, false).unwrap());
                }
            })
        });

        group.throughput(Throughput::Bytes(total_decoded_bytes as u64));
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

/// The chunked decode bench across two vocabulary sizes.
///
/// Decode's cost is one random probe per token into an index sized by the *id space*, so the
/// vocabulary's size is a first-order variable and one model cannot show it. llama-3 is 128k ids
/// (a ~513 kB index); gemma is 262k (~1.05 MB), with more distinct ids live per corpus. A change
/// that helps one and not the other is a cache-residency effect, and this is what says so.
pub fn decode_by_vocab(c: &mut Criterion) {
    for (model, config) in [
        // ByteLevel, 128k ids -> ~513 kB index.
        ("llama3", "../data/llama-3-tokenizer.json"),
        // ByteLevel, 200k ids -> ~800 kB index. The residency control: same decode path as
        // llama-3, 1.56x the table.
        ("gptoss", "../data/gpt-oss.json"),
        // NOT ByteLevel: BPE + byte_fallback with a Replace/ByteFallback/Fuse decoder chain, so
        // this one does not take the fast path at all and shows what the generic route costs.
        ("gemma", "../data/gemma-4.json"),
    ] {
        let tokenizer = load(config);
        for (corpus, path) in [
            ("en", "../data/big.txt"),
            ("ja", "../data/unigram_wagahaiwa_nekodearu.txt"),
        ] {
            let data = std::fs::read_to_string(path).unwrap();
            let chunked = encode_chunks(&tokenizer, &data, CHUNK_BYTES);
            let decoded_bytes: usize = chunked
                .iter()
                .map(|ids| tokenizer.decode(ids, false).unwrap().len())
                .sum();
            let tokens: usize = chunked.iter().map(Vec::len).sum();

            let mut group = c.benchmark_group(format!("decode-vocab-{model}-{corpus}"));
            group.sampling_mode(criterion::SamplingMode::Flat);
            group.sample_size(20);
            group.throughput(Throughput::Bytes(decoded_bytes as u64));
            group.bench_function(
                format!(
                    "10 kB chunks, {tokens} tok, {:.2} B/tok",
                    decoded_bytes as f64 / tokens as f64
                ),
                |bencher| {
                    bencher.iter(|| {
                        for ids in &chunked {
                            black_box(tokenizer.decode(ids, false).unwrap());
                        }
                    })
                },
            );
        }
    }
}

criterion_group!(decode_benches, decode, decode_by_vocab);
criterion_main!(decode_benches);
