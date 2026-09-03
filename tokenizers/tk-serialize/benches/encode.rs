//! Encode, and where the time goes inside it.
//!
//! Two fused rows for the whole pipeline, then one row per stage, so a regression can be
//! attributed to a stage instead of just observed. Deliberately small: the model matrix, the
//! per-language sweeps and the cross-engine comparisons live in huggingface/tokbench.
//!
//! Reading the numbers: the stage rows are single-chunk, single-threaded and take a `&str`, so
//! they undershoot the fused rows and do not sum to them. The gap is the fusion, the threading,
//! the post-processor and the `Inputs` copy that the public `encode` signature requires -- not
//! measurement error. Compare a stage against itself across commits, and the fused rows against
//! each other.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use tk_encode::pipeline::{
    Model, PipelineModel, PipelineTokenizer, PreTokenizer, PreTokenizerScratch, Span,
    SpecialSegmentIterator, normalize_all,
};

const BATCH_SIZE: usize = 1_000;

/// `deepseek-v4` because it is the one config in `../data` that exercises every stage this bench
/// can reach: a real pre-tokenizer that actually splits English, a BPE model, and 1283 added
/// tokens, which is what makes its added-token scan worth timing. Its normalizer is an empty
/// `Sequence`, so that row is skipped -- see the guard below. Add fixtures here to widen the
/// sweep; the rows are emitted per fixture.
const FIXTURES: [(&str, &str); 1] = [("encode-deepseek", "../data/deepseek-v4.json")];

/// Read a real config from `../data`. They are all still version `1.0`, so run the upgrade pass
/// first -- the reader only accepts canonical `2.0`.
fn load(path: &str) -> PipelineTokenizer {
    let canonical = tk_convert::canonicalize_file(path).unwrap();
    tk_serialize::from_json(&canonical).unwrap()
}

fn pre_tokenize(tokenizer: &PipelineTokenizer, text: &str, spans: &mut Vec<Span>) {
    let mut scratch = PreTokenizerScratch::default();
    spans.clear();
    tokenizer
        .get_pre_tokenizer()
        .pre_tokenize(text, &mut scratch, spans)
        .unwrap();
}

pub fn encode(c: &mut Criterion) {
    let data = std::fs::read_to_string("../data/big.txt").unwrap();
    let lines: Vec<&str> = data.lines().collect();
    let batches: Vec<Vec<&str>> = lines.chunks(BATCH_SIZE).map(<[&str]>::to_vec).collect();

    for (name, path) in FIXTURES {
        let tokenizer = load(path);

        let mut group = c.benchmark_group(name);
        group.sampling_mode(criterion::SamplingMode::Flat);
        group.sample_size(20);
        group.throughput(Throughput::Bytes(data.len() as u64));

        // ---- the whole thing ----
        group.bench_function("fused", |b| {
            b.iter(|| {
                for line in &lines {
                    black_box(tokenizer.encode(*line, false).wait().unwrap());
                }
            })
        });
        group.bench_function("fused-batch", |b| {
            b.iter(|| {
                for batch in &batches {
                    black_box(tokenizer.encode(batch.as_slice(), false).wait().unwrap());
                }
            })
        });

        // ---- stage 1: the added-token scan, over the raw input ----
        group.bench_function("stage/added-tokens", |b| {
            b.iter(|| {
                for line in &lines {
                    let segments =
                        SpecialSegmentIterator::new(line, tokenizer.get_added_vocabulary(), false);
                    for segment in segments {
                        black_box(&segment);
                    }
                }
            })
        });

        // ---- stage 2: normalization ----
        // Skipped when the config has no normalizer: `normalize_all` over an empty slice returns
        // the input borrowed, and a row reporting the cost of one function call is just noise.
        // Every byte-level BPE in `../data` is in that boat; `llama-2.json` is one that is not.
        if !tokenizer.get_normalizers().is_empty() {
            group.bench_function("stage/normalize", |b| {
                b.iter(|| black_box(normalize_all(tokenizer.get_normalizers(), &data).unwrap()))
            });
        }

        // Every later stage reads normalized text, so normalize once here rather than inside each.
        let normalized = normalize_all(tokenizer.get_normalizers(), &data)
            .unwrap()
            .into_owned();

        // ---- stage 3: pre-tokenization ----
        // Guarded like `normalize`: a config can have no pre-tokenizer at all (`llama-2.json`),
        // and then this times one function call rather than any work.
        let mut spans = Vec::new();
        pre_tokenize(&tokenizer, &normalized, &mut spans);
        if !spans.is_empty() {
            group.bench_function("stage/pre-tokenize", |b| {
                let mut spans = Vec::new();
                b.iter(|| {
                    pre_tokenize(&tokenizer, &normalized, &mut spans);
                    black_box(spans.len())
                })
            });
        }

        // ---- stage 4: the model, one call per pre-token ----
        let words: Vec<&str> = spans.iter().map(|span| &normalized[span.range()]).collect();

        // `tokenize_pipeline` lives on the concrete model, not on the `PipelineModel` enum, so the
        // row exists per model kind. Both fixtures here are BPE.
        // Irrefutable in a `bpe`-only build, where `PipelineModel` has exactly one variant.
        #[allow(irrefutable_let_patterns)]
        if let PipelineModel::BPE(bpe) = tokenizer.get_model() {
            group.bench_function("stage/model", |b| {
                let mut scratch = bpe.init_scratch();
                let mut out = Vec::new();
                b.iter(|| {
                    out.clear();
                    for word in &words {
                        bpe.tokenize_pipeline(word, &mut scratch, &mut out).unwrap();
                    }
                    black_box(out.len())
                })
            });
        }

        // No `stage/post-process` row: `PipelineTokenizer::post_process` is private and
        // `PipelinePostProcessor` exposes only its templates, so that stage shows up today only
        // inside the fused rows.
    }
}

criterion_group!(encode_benches, encode);
criterion_main!(encode_benches);
