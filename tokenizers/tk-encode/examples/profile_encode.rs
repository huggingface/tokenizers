//! Single-thread E2E encode profiler for one model on one fixture.
//!
//! `fixture_bench.rs` answers *how fast*; this answers *where the time goes*. The whole
//! measured region sits under one `#[inline(never)]` frame (`hot_loop`), so a sampled
//! profile can be collapsed to just the encode path — model load, JSON parsing and
//! fixture I/O stay outside it and can be filtered out wholesale.
//!
//! Chunking matches `fixture_bench` phase 1 (10 kB inputs, ≤100 of them, warm cache,
//! `add_special_tokens = true`) so the profile describes the same regime the headline
//! MB/s numbers come from. Unlike `fixture_bench` no synthetic added tokens are
//! injected — the model is profiled exactly as shipped.
//!
//! ```text
//! cargo build --release --example profile_encode
//! samply record --main-thread-only -r 4000 -o prof.json.gz \
//!     -- ./target/release/examples/profile_encode data/gpt2.json data/fixtures/lang/eng_Latn.txt
//! ```
//!
//! `--stages` instead prints the `encode_generic` ablation ladder (ns/byte per stage)
//! for the same model/fixture pair, as the quantitative companion to the flamegraph.
//! Keep it out of profiled runs: partial-stage passes would land in the same samples.

use std::convert::TryFrom;
use std::hint::black_box;
use std::time::Instant;

use tk_encode::pipeline::{Model, PipelineTokenizer};
use tk_encode::{ModelWrapper, Tokenizer};

const CHUNK_BYTES: usize = 10 * 1024;
const MAX_CHUNKS: usize = 100;
const DEFAULT_SECONDS: f64 = 6.0;
const REPS: usize = 5;

fn make_chunks(text: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut cur = String::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        if !cur.is_empty() {
            cur.push('\n');
        }
        cur.push_str(line);
        if cur.len() >= CHUNK_BYTES {
            chunks.push(std::mem::take(&mut cur));
            if chunks.len() == MAX_CHUNKS {
                return chunks;
            }
        }
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

/// Encode every chunk in a loop until `seconds` have elapsed; returns (passes, tokens).
///
/// `inline(never)` is the whole point: this frame is the filter the flamegraph post-pass
/// keys on, so setup can be discarded without guessing which symbols belong to encode.
#[inline(never)]
fn hot_loop(pipeline: &PipelineTokenizer, chunks: &[String], seconds: f64) -> (usize, usize) {
    let start = Instant::now();
    let mut passes = 0usize;
    let mut tokens = 0usize;
    while start.elapsed().as_secs_f64() < seconds {
        for chunk in chunks {
            tokens += pipeline.encode(chunk, true).unwrap().len();
        }
        passes += 1;
    }
    black_box(tokens);
    (passes, tokens)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Median seconds for one pass over `chunks` through `encode_generic::<STAGE>`, on a
/// caller-owned scratch reused across chunks. Mirrors `fixture_bench::stage_secs`.
fn stage_secs<const STAGE: u8>(pipeline: &PipelineTokenizer, chunks: &[String]) -> f64 {
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
    run();
    let mut samples = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t = Instant::now();
        run();
        samples.push(t.elapsed().as_secs_f64());
    }
    median(samples)
}

/// Order-sensitive hash of every id one pass emits, for diffing two builds of the encoder.
fn checksum(pipeline: &PipelineTokenizer, chunks: &[String]) -> (u64, usize) {
    let mut h: u64 = 0xcbf29ce484222325;
    let mut n = 0;
    for chunk in chunks {
        for token in pipeline.encode(chunk, true).unwrap() {
            h = (h ^ token.id as u64).wrapping_mul(0x100000001b3);
            n += 1;
        }
    }
    (h, n)
}

/// Pre-token count for one pass, so per-span costs can be quoted per word rather than as a
/// share. `STAGE_SPLIT` stops after the pre-tokenizer, leaving its spans in `pre_tokens`.
fn count_spans(pipeline: &PipelineTokenizer, chunks: &[String]) -> usize {
    let mut out = Vec::new();
    let mut pre_tokens = Vec::new();
    let mut scratch = pipeline.get_model().init_scratch();
    let mut spans = 0;
    for chunk in chunks {
        out.clear();
        pre_tokens.clear();
        let _ = pipeline.encode_generic::<{ PipelineTokenizer::STAGE_SPLIT }>(
            chunk,
            true,
            &mut pre_tokens,
            &mut scratch,
            &mut out,
        );
        spans += pre_tokens.len();
    }
    spans
}

fn print_stages(pipeline: &PipelineTokenizer, chunks: &[String], bytes: usize) {
    let t_frame = stage_secs::<{ PipelineTokenizer::STAGE_FRAME }>(pipeline, chunks);
    let t_norm = stage_secs::<{ PipelineTokenizer::STAGE_NORMALIZE }>(pipeline, chunks);
    let t_split = stage_secs::<{ PipelineTokenizer::STAGE_SPLIT }>(pipeline, chunks);
    let t_model = stage_secs::<{ PipelineTokenizer::STAGE_MODEL }>(pipeline, chunks);
    let t_post = stage_secs::<{ PipelineTokenizer::STAGE_POSTPROCESS }>(pipeline, chunks);
    let nspb = |s: f64| s * 1e9 / bytes as f64;
    println!("{{");
    println!("  \"added_split\": {:.4},", nspb(t_frame));
    println!("  \"pre_tokenize\": {:.4},", nspb(t_split - t_frame));
    println!("  \"normalize\": {:.4},", nspb(t_norm - t_split));
    println!("  \"model\": {:.4},", nspb(t_model - t_norm));
    println!("  \"post_process\": {:.4},", nspb(t_post - t_model));
    println!("  \"total\": {:.4}", nspb(t_post));
    println!("}}");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: profile_encode <model.json> <fixture.txt> [seconds|--stages]");
        std::process::exit(2);
    }
    let stages = args.iter().any(|a| a == "--stages");
    let seconds = args
        .get(3)
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(DEFAULT_SECONDS);

    let mut tok = Tokenizer::from_file(&args[1]).expect("load tokenizer");
    // `PipelineBPE::from_bpe` filters out a zero capacity, so this is the off switch for the
    // BPE word cache -- the A/B that shows what the cache probe is worth on a given model.
    if args.iter().any(|a| a == "--no-cache")
        && let ModelWrapper::BPE(bpe) = tok.get_model_mut()
    {
        bpe.resize_cache(0);
    }
    let pipeline = PipelineTokenizer::try_from(&tok).expect("build pipeline");
    let text = std::fs::read_to_string(&args[2]).expect("read fixture");
    let chunks = make_chunks(&text);
    let bytes: usize = chunks.iter().map(String::len).sum();

    // Warm-up: fills the BPE word cache from this corpus, so the measured region is the
    // warm steady state a plain `.encode()` loop reaches rather than cache cold-start.
    for chunk in &chunks {
        black_box(pipeline.encode(chunk, true).unwrap().len());
    }

    if args.iter().any(|a| a == "--checksum") {
        let (h, n) = checksum(&pipeline, &chunks);
        println!("{{\"ids\": {n}, \"checksum\": \"{h:016x}\"}}");
        return;
    }
    if args.iter().any(|a| a == "--spans") {
        let spans = count_spans(&pipeline, &chunks);
        println!(
            "{{\"bytes\": {bytes}, \"spans\": {spans}, \"bytes_per_span\": {:.3}}}",
            bytes as f64 / spans as f64
        );
        return;
    }
    if stages {
        print_stages(&pipeline, &chunks, bytes);
        return;
    }

    let t = Instant::now();
    let (passes, tokens) = hot_loop(&pipeline, &chunks, seconds);
    let elapsed = t.elapsed().as_secs_f64();
    let mbps = (bytes * passes) as f64 / elapsed / 1e6;
    eprintln!(
        "{} on {}: {bytes} B x {passes} passes in {elapsed:.2}s = {mbps:.1} MB/s ({tokens} tokens)",
        args[1], args[2]
    );
}
