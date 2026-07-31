//! One end-to-end throughput number for one model on one corpus, for A/B runs
//! between two builds of this crate (fold on vs fold off, run alternately at
//! process level so binary-layout noise averages out).
//!
//! Mirrors `fixture_bench`'s throughput phase: single thread, ~10 kB chunks,
//! `add_special_tokens` on, one warm-up pass to fill the caches, then the
//! median of the timed passes.
//!
//!     cargo run --release --example fold_ab -- <tokenizer.json> <corpus.txt> [passes]

use std::convert::TryFrom;
use std::time::Instant;

use tk_encode::Tokenizer;
use tk_encode::pipeline::PipelineTokenizer;

const CHUNK: usize = 10_000;
const CORPUS_CAP: usize = 4 << 20;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, model, corpus] = &args[..3] else {
        eprintln!("usage: fold_ab <tokenizer.json> <corpus.txt> [passes]");
        std::process::exit(2);
    };
    let passes: usize = args.get(3).map_or(5, |p| p.parse().unwrap());

    let tok = Tokenizer::from_file(model).unwrap();
    let pipeline = PipelineTokenizer::try_from(&tok).unwrap();

    let mut text = std::fs::read_to_string(corpus).unwrap();
    if text.len() > CORPUS_CAP {
        let mut cap = CORPUS_CAP;
        while !text.is_char_boundary(cap) {
            cap -= 1;
        }
        text.truncate(cap);
    }
    let chunks: Vec<&str> = {
        let mut chunks = Vec::new();
        let mut rest = text.as_str();
        while !rest.is_empty() {
            let mut cut = CHUNK.min(rest.len());
            while !rest.is_char_boundary(cut) {
                cut -= 1;
            }
            let (head, tail) = rest.split_at(cut);
            chunks.push(head);
            rest = tail;
        }
        chunks
    };
    let bytes: usize = chunks.iter().map(|c| c.len()).sum();

    let encode_all = || {
        let mut ids = 0usize;
        for chunk in &chunks {
            ids += pipeline.encode(chunk, true).unwrap().len();
        }
        ids
    };

    let ids = encode_all(); // warm-up: fills the scratch pool and the word cache
    let mut timings: Vec<f64> = (0..passes)
        .map(|_| {
            let start = Instant::now();
            encode_all();
            start.elapsed().as_secs_f64()
        })
        .collect();
    timings.sort_by(f64::total_cmp);
    let median = timings[timings.len() / 2];

    println!(
        "{:.3} MB/s ({} B, {} ids, {passes} passes)",
        bytes as f64 / median / 1e6,
        bytes,
        ids,
    );
}
