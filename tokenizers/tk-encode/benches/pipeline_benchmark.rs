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

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, Throughput};
use tk_encode::pipeline::PipelineTokenizer;
use tk_encode::Tokenizer;

// (name, tokenizer.json): bert-wiki exercises the Bert/WordPiece path; dsv4 exercises the deepseek
// pre-tokenizer unroll (`fsm_deepseek`) + BPE end-to-end.
const TOKENIZERS: &[(&str, &str)] = &[
    ("bert", "../data/bert-wiki.json"),
    ("dsv4", "../data/deepseek-v4-flash-base-tokenizer.json"),
];

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

// Fixed size (amortized regime) at which the per-stage decomposition is measured.
const STAGE_CHUNK: usize = 10 * 1024;

/// Cumulative-stage ladder: bench `encode_generic::<STAGE>` (every stage up to and
/// including `STAGE`) over `chunks`, reusing the caller-owned buffers. Run for each
/// level and read the stage-to-stage delta as that stage's marginal cost; `POST` −
/// `MODEL` is the post-processor. `add_special_tokens = true` so the post-processor
/// actually runs (framing / special tokens) rather than being a no-op.
fn bench_stage<const STAGE: u8>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    label: &str,
    pipeline: &PipelineTokenizer,
    chunks: &[String],
) {
    group.bench_function(label, |b| {
        let mut out = Vec::new();
        let mut pre = Vec::new();
        b.iter(|| {
            for chunk in chunks {
                out.clear();
                let _ = pipeline.encode_generic::<STAGE>(chunk, true, &mut out, &mut pre);
                black_box(&out);
            }
        });
    });
}

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
        let Ok(oracle) = Tokenizer::from_file(tok_path) else {
            eprintln!("pipeline bench: skip {tok_name} — {tok_path} not found");
            continue;
        };
        let pipeline = match PipelineTokenizer::try_from(&oracle) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("pipeline bench: skip {tok_name} — not pipeline-supported: {e}");
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
                        let mut n = 0usize;
                        for chunk in &chunks {
                            n += pipeline.encode(chunk, false).unwrap().len();
                        }
                        black_box(n)
                    })
                });
            }
            group.finish();

            // Per-stage decomposition at a fixed size: each cumulative level, so the
            // stage-to-stage throughput delta isolates that stage's cost. `5-post` is the
            // post-processor stage (the one suspected to be slow); it is a no-op — equal to
            // `4-model` — for tokenizers without a post-processor.
            let chunks = make_chunks(&lines, STAGE_CHUNK);
            let total_bytes: u64 = chunks.iter().map(|s| s.len() as u64).sum();
            let mut sg = c.benchmark_group(format!("{tok_name}-{corpus}-stages"));
            sg.throughput(Throughput::Bytes(total_bytes));
            bench_stage::<{ PipelineTokenizer::STAGE_FRAME }>(&mut sg, "1-frame", &pipeline, &chunks);
            bench_stage::<{ PipelineTokenizer::STAGE_NORMALIZE }>(
                &mut sg,
                "2-normalize",
                &pipeline,
                &chunks,
            );
            bench_stage::<{ PipelineTokenizer::STAGE_SPLIT }>(&mut sg, "3-split", &pipeline, &chunks);
            bench_stage::<{ PipelineTokenizer::STAGE_MODEL }>(&mut sg, "4-model", &pipeline, &chunks);
            bench_stage::<{ PipelineTokenizer::STAGE_POST }>(&mut sg, "5-post", &pipeline, &chunks);
            sg.finish();
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
