use std::time::Duration;

use criterion::{Criterion, Throughput};
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::{decoders::wordpiece::WordPiece as WordPieceDecoder, Tokenizer};

use crate::common::{iter_bench_decode, iter_bench_decode_batch, iter_bench_decode_stream};

#[macro_use]
extern crate criterion;

mod common;

const BATCH_SIZE: usize = 1_000;
// How many tokens to run through streaming decode pipeline
const STREAM_TOKEN_BUDGET: usize = 300_000;

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

pub fn decode(c: &mut Criterion) {
    let big_txt = std::fs::read_to_string("data/big.txt").unwrap();
    let japanese_txt = std::fs::read_to_string("data/unigram_wagahaiwa_nekodearu.txt").unwrap();
    let pipelines = [
        (
            "decode-llama3-en",
            Tokenizer::from_file("data/llama-3-tokenizer.json").unwrap(),
            &big_txt,
        ),
        (
            "decode-llama3-ja",
            Tokenizer::from_file("data/llama-3-tokenizer.json").unwrap(),
            &japanese_txt,
        ),
        (
            "decode-albert",
            Tokenizer::from_file("data/albert-base-v1-tokenizer.json").unwrap(),
            &big_txt,
        ),
        (
            "decode-bert",
            {
                let mut tokenizer = Tokenizer::new(
                    WordPiece::from_file("data/bert-base-uncased-vocab.txt")
                        .build()
                        .unwrap(),
                );
                tokenizer
                    .with_normalizer(Some(BertNormalizer::default()))
                    .unwrap();
                tokenizer.with_pre_tokenizer(Some(BertPreTokenizer));
                tokenizer.with_decoder(Some(WordPieceDecoder::default()));
                tokenizer
            },
            &big_txt,
        ),
    ];

    for (name, tokenizer, data) in pipelines {
        // Build input: tokenize data and group it in lines and batches
        let mut lines: Vec<Vec<u32>> = vec![];
        for line in data.lines() {
            let encoded: Vec<u32> = tokenizer
                .encode_fast(line, false)
                .unwrap()
                .get_ids()
                .to_vec();
            lines.push(encoded);
        }
        // Long lines, for decode_stream
        let fused_lines: Vec<Vec<u32>> = lines
            .chunks(BATCH_SIZE)
            .map(|chunk| chunk.concat())
            .collect();
        let batches: Vec<Vec<&[u32]>> = lines
            .chunks(BATCH_SIZE)
            .map(|chunk| chunk.iter().map(Vec::as_slice).collect())
            .collect();

        let total_decoded_bytes = {
            let mut length = 0;
            for line in &lines {
                length += tokenizer.decode(line, false).unwrap().len();
            }
            length
        };

        let mut group = c.benchmark_group(name);
        group.sampling_mode(criterion::SamplingMode::Flat);

        group.throughput(Throughput::Bytes(total_decoded_bytes as u64));
        group.bench_function("decode", |bencher| {
            bencher.iter_custom(|num_iterations| {
                iter_bench_decode(num_iterations, &tokenizer, &lines)
            });
        });
        group.bench_function("decode_batch", |bencher| {
            bencher.iter_custom(|num_iterations| {
                iter_bench_decode_batch(num_iterations, &tokenizer, &batches)
            });
        });

        let stream_lines = take_token_budget(&lines, STREAM_TOKEN_BUDGET);
        let stream_fused = take_token_budget(&fused_lines, STREAM_TOKEN_BUDGET);

        // Throughput expressed in tokens / sec
        group.throughput(Throughput::Elements(
            stream_lines.iter().map(|line| line.len()).sum::<usize>() as u64,
        ));
        group.bench_function("decode_stream, short lines", |bencher| {
            bencher.iter_custom(|num_iterations| {
                iter_bench_decode_stream(num_iterations, &tokenizer, &stream_lines)
            })
        });
        // Throughput expressed in tokens / sec
        group.throughput(Throughput::Elements(
            stream_fused.iter().map(|line| line.len()).sum::<usize>() as u64,
        ));
        group.bench_function("decode_stream, long lines", |bencher| {
            bencher.iter_custom(|num_iterations| {
                iter_bench_decode_stream(num_iterations, &tokenizer, &stream_fused)
            })
        });
    }
}

criterion_group!(
    name = decode_benches;
    config = Criterion::default().sample_size(20).measurement_time(Duration::new(10, 0));
    targets = decode
);

criterion_main!(decode_benches);
