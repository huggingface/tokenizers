//! Per-stage cost of the encode path, measured so that a stateful word cache cannot corrupt it.
//!
//! `encode_generic_into` takes STAGE as a const generic, so each level is a separately
//! monomorphised function with no runtime gate. Timing the cumulative ladder and differencing
//! gives the per-stage cost:
//!
//!   FRAME       segment iteration + the call itself
//!   NORMALIZE   + normalizers
//!   SPLIT       + pre-tokenization
//!   MODEL       + fold / cache / merge
//!   POSTPROCESS + prefix/suffix
//!
//! Two traps this avoids, both of which produced nonsense first:
//!
//! 1. **Re-encoding the same text.** Taking the best of N passes over one slice measures the word
//!    cache memoizing that exact text -- it read gpt2/english at 344 MB/s where tokbench, on the
//!    same code, says 199. So the corpus is cut into `SLICES + 1` disjoint slices, tokbench-style:
//!    slice 0 warms, each timed pass sees text it has never seen, and the median is reported.
//!
//! 2. **A cache shared between stage levels.** Running level 3 and then level 4 over the same
//!    slices leaves level 4 with a cache level 3 already filled, which made "model" cost more than
//!    "total" (shares of 281%). So every level gets a FRESH tokenizer, hence a fresh cache.
//!
//! Load happens outside the timer.
use std::hint::black_box;
use std::time::Instant;

use tk_encode::Tokenizer;
use tk_encode::pipeline::PipelineTokenizer;

const CHUNK: usize = 10 * 1024;
const MAX_CHUNKS: usize = 100;
const SLICES: usize = 5;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model = args.next().expect("usage: stage_split <model.json> <corpus.txt>...");
    let legacy = Tokenizer::from_file(&model).expect("load model");

    println!(
        "{:<12} {:>7} {:>7} {:>7} {:>7} {:>7}  {:>8}   {}",
        "corpus", "frame", "norm", "split", "model", "post", "total", "split% / model%   MB/s"
    );
    for path in args {
        let text = std::fs::read_to_string(&path).expect("read corpus");
        let mut end = (CHUNK * MAX_CHUNKS).min(text.len());
        while end < text.len() && !text.is_char_boundary(end) {
            end += 1;
        }
        let text = &text[..end];
        let mut chunks: Vec<&str> = Vec::new();
        let mut s = 0;
        while s < text.len() {
            let mut e = (s + CHUNK).min(text.len());
            while e < text.len() && !text.is_char_boundary(e) {
                e += 1;
            }
            chunks.push(&text[s..e]);
            s = e;
        }
        // SLICES + 1 disjoint slices, round-robin so each is a comparable mix of the corpus.
        let slices: Vec<Vec<&str>> = (0..=SLICES)
            .map(|s| chunks.iter().skip(s).step_by(SLICES + 1).copied().collect())
            .collect();

        let mut cumulative = [0.0f64; 5];
        for (level, slot) in cumulative.iter_mut().enumerate() {
            // Fresh tokenizer per level => fresh word cache, so no level inherits another's.
            let pipe = PipelineTokenizer::try_from(&legacy).expect("build pipeline");
            let mut out = Vec::with_capacity(1 << 20);
            let mut run = |slice: &[&str]| -> f64 {
                let bytes: usize = slice.iter().map(|c| c.len()).sum();
                let start = Instant::now();
                for ch in slice {
                    out.clear();
                    match level {
                        0 => pipe.encode_generic_into::<{ PipelineTokenizer::STAGE_FRAME }>(ch, false, &mut out),
                        1 => pipe.encode_generic_into::<{ PipelineTokenizer::STAGE_NORMALIZE }>(ch, false, &mut out),
                        2 => pipe.encode_generic_into::<{ PipelineTokenizer::STAGE_SPLIT }>(ch, false, &mut out),
                        3 => pipe.encode_generic_into::<{ PipelineTokenizer::STAGE_MODEL }>(ch, false, &mut out),
                        _ => pipe.encode_generic_into::<{ PipelineTokenizer::STAGE_POSTPROCESS }>(ch, false, &mut out),
                    }
                    .unwrap();
                    black_box(&out);
                }
                start.elapsed().as_secs_f64() * 1e9 / bytes as f64
            };
            run(&slices[0]); // warm-up, untimed
            *slot = median(slices[1..].iter().map(|s| run(s)).collect());
        }

        let frame = cumulative[0];
        let norm = (cumulative[1] - cumulative[0]).max(0.0);
        let split = (cumulative[2] - cumulative[1]).max(0.0);
        let model = (cumulative[3] - cumulative[2]).max(0.0);
        let post = (cumulative[4] - cumulative[3]).max(0.0);
        let total = cumulative[4];
        let name = std::path::Path::new(&path).file_stem().unwrap().to_string_lossy();
        println!(
            "{name:<12} {frame:>7.2} {norm:>7.2} {split:>7.2} {model:>7.2} {post:>7.2}  {total:>8.2}   {:>5.0}% / {:>4.0}%   {:>6.0}",
            split / total * 100.0,
            model / total * 100.0,
            1000.0 / total,
        );
    }
}
