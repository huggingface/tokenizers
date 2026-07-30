//! Where the pre-tokenize budget goes, per stage, for the gpt2 (ByteLevel) split.
//!
//! ```text
//!   cargo run --release --example split_stages
//! ```
//!
//! The stages are the ones `PipelineTokenizer` runs, in order, and each row is
//! cumulative over the row above it, so a stage's own cost is the difference
//! between the two. The last row adds what the model does before it can use a
//! span: build the word-cache key for it.
//!
//! ns/byte, so the numbers line up with the per-byte breakdowns in the profiles.

use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use atomsplit::classify::classify;
use atomsplit::fsm::{Span, fsm_byte_level};
use tk_encode::utils::word_cache::WordCache;

const DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data");
const CHUNK_BYTES: usize = 10 * 1024;
const MAX_CHUNKS: usize = 100;
const REPS: usize = 9;

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

fn ns_per_byte(bytes: usize, mut run: impl FnMut() -> usize) -> f64 {
    run();
    let mut samples: Vec<f64> = (0..REPS)
        .map(|_| {
            let start = Instant::now();
            black_box(run());
            start.elapsed().as_secs_f64()
        })
        .collect();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[REPS / 2] * 1e9 / bytes as f64
}

fn main() {
    println!("fixture\tB/word\ttags\t+fsm\t+copy\t+keys");
    for (name, file) in FIXTURES {
        let text = std::fs::read_to_string(Path::new(DATA_DIR).join(file)).unwrap();
        let chunks = make_chunks(&text);
        let bytes: usize = chunks.iter().map(String::len).sum();

        let widest = chunks.iter().map(String::len).max().unwrap();
        let mut tags = vec![0u8; widest];
        let mut spans = vec![Span::default(); widest + 1];
        let mut out: Vec<Span> = Vec::with_capacity(widest + 1);
        let cache = WordCache::new(1 << 16);

        let tags_only = ns_per_byte(bytes, || {
            let mut n = 0;
            for chunk in &chunks {
                let b = chunk.as_bytes();
                classify(b, &mut tags[..b.len()]);
                n += tags[b.len() - 1] as usize;
            }
            n
        });
        let with_fsm = ns_per_byte(bytes, || {
            let mut n = 0;
            for chunk in &chunks {
                let b = chunk.as_bytes();
                classify(b, &mut tags[..b.len()]);
                n += fsm_byte_level(b, &tags[..b.len()], &mut spans[..b.len() + 1]);
            }
            n
        });
        let with_copy = ns_per_byte(bytes, || {
            let mut n = 0;
            for chunk in &chunks {
                let b = chunk.as_bytes();
                classify(b, &mut tags[..b.len()]);
                let k = fsm_byte_level(b, &tags[..b.len()], &mut spans[..b.len() + 1]);
                out.clear();
                out.extend_from_slice(&spans[..k]);
                n += out.len();
            }
            n
        });
        let with_keys = ns_per_byte(bytes, || {
            let mut n = 0;
            for chunk in &chunks {
                let b = chunk.as_bytes();
                classify(b, &mut tags[..b.len()]);
                let k = fsm_byte_level(b, &tags[..b.len()], &mut spans[..b.len() + 1]);
                out.clear();
                out.extend_from_slice(&spans[..k]);
                for span in &out {
                    n += !cache.key(&b[span.range()]).is_none() as usize;
                }
            }
            n
        });

        let words: usize = chunks
            .iter()
            .map(|chunk| {
                let b = chunk.as_bytes();
                classify(b, &mut tags[..b.len()]);
                fsm_byte_level(b, &tags[..b.len()], &mut spans[..b.len() + 1])
            })
            .sum();

        println!(
            "{name}\t{:.1}\t{tags_only:.3}\t{with_fsm:.3}\t{with_copy:.3}\t{with_keys:.3}",
            bytes as f64 / words as f64,
        );
    }
}
