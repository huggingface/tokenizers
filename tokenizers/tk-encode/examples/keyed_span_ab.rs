//! End-to-end throughput of [`PipelineTokenizer::encode`] on a fixed set of
//! (model, fixture) pairs, for comparing two builds of the crate.
//!
//! ```text
//!   cargo run --release --example keyed_span_ab
//!   cargo run --release --example keyed_span_ab -- --reps 9
//! ```
//!
//! One line per pair: `model fixture MB/s(median) MB/s(min) MB/s(max) checksum`.
//!
//! # How to read two runs against each other
//!
//! This harness prints numbers for one build. A difference between two builds is
//! only a result if it beats what the *same* code shows when built twice, which
//! on an M3 Max has been worth up to 7% from binary layout alone. So: build the
//! unchanged crate in two separate worktrees plus the changed one in a third, run
//! all three alternately, and read the unchanged pair's gap as the noise floor.
//!
//! The `checksum` column is order-sensitive over the whole id stream. Two builds
//! that disagree on it are not comparable, whatever their throughput says.
//!
//! Every pair starts from a cold tokenizer, so the word cache fills from this
//! corpus and no other — then one warm-up pass, then the timed reps. That is the
//! state a plain `.encode()` loop over a corpus reaches.

use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use tk_encode::Tokenizer;
use tk_encode::pipeline::PipelineTokenizer;

const DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data");
const CHUNK_BYTES: usize = 10 * 1024;
const MAX_CHUNKS: usize = 100;

/// Models whose pre-tokenizer cuts the text into words, so the word cache is on
/// the path: gpt2 splits on the ByteLevel regex, llama-3 on the cl100k one and
/// takes whole-word vocabulary hits (`ignore_merges`) on top.
const MODELS: [(&str, &str); 2] = [
    ("gpt2", "gpt2.json"),
    ("llama-3", "llama-3-tokenizer.json"),
];

/// One script per writing system the split behaves differently on: English words
/// are long and repeat, agentic code is mostly punctuation and indentation, and
/// Chinese has no spaces at all.
const FIXTURES: [(&str, &str); 3] = [
    ("eng_Latn", "fixtures/lang/eng_Latn.txt"),
    ("agentic_swe", "fixtures/modalities/agentic_swe.txt"),
    ("cmn_Hani", "fixtures/lang/cmn_Hani.txt"),
];

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

/// FNV-1a over the id stream, so a build that tokenizes differently cannot pass
/// as a faster one.
fn checksum(ids: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for id in ids {
        for byte in id.to_le_bytes() {
            h = (h ^ byte as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let reps: usize = match args.iter().position(|a| a == "--reps") {
        Some(i) => args[i + 1].parse().expect("--reps takes a number"),
        None => 5,
    };

    for (model_name, model_file) in MODELS {
        let tokenizer = Tokenizer::from_file(Path::new(DATA_DIR).join(model_file)).unwrap();
        for (fixture_name, fixture_file) in FIXTURES {
            let text = std::fs::read_to_string(Path::new(DATA_DIR).join(fixture_file)).unwrap();
            let chunks = make_chunks(&text);
            let bytes: usize = chunks.iter().map(String::len).sum();

            // Rebuilt per fixture: a fresh pipeline is a fresh word cache.
            let pipeline = PipelineTokenizer::try_from(&tokenizer).unwrap();
            let mut ids = Vec::new();
            let mut pass = || {
                ids.clear();
                for chunk in &chunks {
                    ids.extend(pipeline.encode(chunk, true).unwrap().iter().map(|t| t.id));
                }
                ids.len()
            };

            pass();
            let mut rates: Vec<f64> = Vec::with_capacity(reps);
            for _ in 0..reps {
                let start = Instant::now();
                black_box(pass());
                rates.push(bytes as f64 / start.elapsed().as_secs_f64() / 1e6);
            }
            rates.sort_by(|a, b| a.partial_cmp(b).unwrap());

            println!(
                "{model_name}\t{fixture_name}\t{:.1}\t{:.1}\t{:.1}\t{:016x}",
                rates[reps / 2],
                rates[0],
                rates[reps - 1],
                checksum(&ids),
            );
        }
    }
}
