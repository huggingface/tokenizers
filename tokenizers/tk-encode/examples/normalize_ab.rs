//! A/B harness for changes to the pipeline's normalize stage: for each tokenizer config given on
//! the command line, times the `encode_generic` ablation ladder over `data/big.txt` and prints the
//! frame and normalize levels plus the full encode throughput.
//!
//!     cargo run --release -p tk-encode --example normalize_ab -- data/llama-2.json data/gemma-4.json
//!
//! Binary layout alone moves numbers by a few percent, so never compare two builds from one
//! process each: build one binary per side, keep both, and alternate runs.

use std::convert::TryFrom;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use tk_encode::Tokenizer;
use tk_encode::pipeline::{Model, PipelineTokenizer};

const DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data");
/// Per-input-overhead is amortized at this size; same regime as `fixture_bench`.
const CHUNK_BYTES: usize = 10 * 1024;
const TOTAL_BYTES: usize = 4 * 1024 * 1024;
const REPS: usize = 9;

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

/// Median seconds of one warm pass over `chunks` at ladder level `STAGE`;
/// same shape as `fixture_bench::stage_secs`.
fn stage_secs<const STAGE: u8>(pipeline: &PipelineTokenizer, chunks: &[&str]) -> f64 {
    let mut out = Vec::new();
    let mut pre_tokens = Vec::new();
    let mut scratch = pipeline.get_model().init_scratch();
    let mut run = || {
        for chunk in chunks {
            out.clear();
            let _ = pipeline.encode_generic::<STAGE>(
                chunk,
                true,
                &mut pre_tokens,
                &mut scratch,
                &mut out,
            );
            black_box(&out);
            black_box(&pre_tokens);
        }
    };
    run(); // warm-up
    let mut samples = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let start = Instant::now();
        run();
        samples.push(start.elapsed().as_secs_f64());
    }
    median(samples)
}

/// The first `TOTAL_BYTES` of big.txt in `CHUNK_BYTES` pieces, cut on char boundaries.
fn chunks_of(text: &str) -> Vec<&str> {
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < text.len().min(TOTAL_BYTES) {
        let mut end = (start + CHUNK_BYTES).min(text.len());
        while !text.is_char_boundary(end) {
            end += 1;
        }
        chunks.push(&text[start..end]);
        start = end;
    }
    chunks
}

fn main() {
    let text = std::fs::read_to_string(Path::new(DATA_DIR).join("big.txt"))
        .expect("data/big.txt (fetch with `make data/big.txt`)");
    let chunks = chunks_of(&text);
    let bytes: usize = chunks.iter().map(|c| c.len()).sum();

    for config in std::env::args().skip(1) {
        let tok = Tokenizer::from_file(&config).expect("tokenizer config");
        let pipeline = PipelineTokenizer::try_from(&tok).expect("pipeline builds");

        let t_frame = stage_secs::<{ PipelineTokenizer::STAGE_FRAME }>(&pipeline, &chunks);
        let t_norm = stage_secs::<{ PipelineTokenizer::STAGE_NORMALIZE }>(&pipeline, &chunks);
        let t_full = stage_secs::<{ PipelineTokenizer::STAGE_POSTPROCESS }>(&pipeline, &chunks);

        let nspb = |secs: f64| secs * 1e9 / bytes as f64;
        println!(
            "{config}: frame {:.3} ns/B, normalize {:.3} ns/B (marginal {:.3}), full {:.3} ns/B = {:.1} MB/s",
            nspb(t_frame),
            nspb(t_norm),
            nspb(t_norm - t_frame),
            nspb(t_full),
            bytes as f64 / t_full / 1e6,
        );
    }
}
