//! Throughput of the experimental `PipelineTokenizer`, per corpus, as a
//! function of input length: corpus lines are packed into documents of
//! ~128 B / 1 kB / 10 kB / 100 kB, sweeping from per-input-overhead-dominated
//! to fully amortized.
//!
//! Correctness (id equivalence with the reference `Tokenizer`) is asserted
//! separately in `tests/pipeline_oracle.rs`.

#[macro_use]
extern crate criterion;

use std::convert::TryFrom;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput};
use tk_encode::pipeline::PipelineTokenizer;
use tk_encode::Tokenizer;

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
    let oracle = Tokenizer::from_file("../data/bert-wiki.json").unwrap();
    let pipeline = PipelineTokenizer::try_from(&oracle).unwrap();

    for (corpus, path) in CORPORA {
        let text = std::fs::read_to_string(path).unwrap();
        let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();

        let mut group = c.benchmark_group(format!("pipeline-{corpus}"));
        for (target_bytes, label) in CHUNK_SIZES {
            let chunks = make_chunks(&lines, *target_bytes);
            let total_bytes: u64 = chunks.iter().map(|s| s.len() as u64).sum();
            group.throughput(Throughput::Bytes(total_bytes));
            group.bench_function(BenchmarkId::from_parameter(label), |b| {
                b.iter(|| {
                    let mut n = 0usize;
                    for chunk in &chunks {
                        n += pipeline.encode(chunk, false).unwrap().len();
                    }
                    black_box(n)
                })
            });
        }
        group.finish();
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
