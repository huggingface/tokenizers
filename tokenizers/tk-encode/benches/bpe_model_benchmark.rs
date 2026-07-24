//! Here I want to benchmark various ways we can run BPE merge.

#[macro_use]
extern crate criterion;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput};
use tk_encode::{
    Tokenizer,
    models::bpe::BpeScratch,
    pipeline::{Model, PipelineModel, PipelineToken, PipelineTokenizer},
};

// We will be testing different voacab / merges.
const TOKENIZERS: &[(&str, &str)] = &[("dsv4", "../data/deepseek-v4-flash-base-tokenizer.json")];

const CORPORA: &[(&str, &str)] = &[
    ("big", "../data/big.txt"),
    ("wagahai", "../data/unigram_wagahaiwa_nekodearu.txt"),
];

const CHUNK_SIZES: &[(usize, &str)] = &[
    (128, "128B"),
    (1024, "1kB"),
    (10 * 1024, "10kB"),
    (100 * 1024, "100kB"),
];

fn make_chunks(lines: &[&str], target_bytes: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut cur = String::new();
    for line in lines {
        if !cur.is_empty() {
            cur.push('\n');
        }
        cur.push_str(line);
        if cur.len() >= target_bytes {
            chunks.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

fn bench_pipeline(c: &mut Criterion) {
    for (tok_name, tok_path) in TOKENIZERS {
        // The oracle will use the old merge,
        let Ok(oracle) = Tokenizer::from_file(tok_path) else {
            eprintln!("pipeline bench: skip {tok_name} — {tok_path} not found");
            continue;
        };
        let pipeline = match PipelineTokenizer::try_from(&oracle) {
            Ok(p) => p,
            _ => {
                eprint!("Failed to init from the oracle");
                continue;
            }
        };
        let model = match pipeline.get_model() {
            PipelineModel::BPE(p) => p,
            _ => {
                eprintln!("Only bpe models are supported");
                continue;
            }
        };
        for (corpus, path) in CORPORA {
            let text = std::fs::read_to_string(path).unwrap();
            let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();

            let mut group = c.benchmark_group(format!("{tok_name}-{corpus}"));
            for (target_bytes, label) in CHUNK_SIZES {
                let chunks = make_chunks(&lines, *target_bytes);
                let total_bytes: u64 = chunks.iter().map(|s| s.len() as u64).sum();
                group.throughput(Throughput::Bytes(total_bytes));
                group.bench_function(BenchmarkId::from_parameter(label), |b| {
                    b.iter(|| {
                        for chunk in &chunks {
                            let mut output =
                                Vec::<PipelineToken>::with_capacity(total_bytes as usize);
                            model
                                .tokenize_pipeline(
                                    chunk.as_str(),
                                    &mut model.init_scratch(),
                                    &mut output,
                                )
                                .unwrap();
                            black_box(output);
                        }
                    })
                });
            }
            group.finish();
        }
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(10));
    targets = bench_pipeline
}
criterion_main!(benches);
