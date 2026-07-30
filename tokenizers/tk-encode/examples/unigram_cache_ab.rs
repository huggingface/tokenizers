//! Times the Unigram model stage alone, the way `PipelineTokenizer` drives it: one
//! `tokenize_pipeline` call per pre-token, one scratch reused for the whole corpus.
//!
//! `PipelineTokenizer` cannot build a real Unigram tokenizer yet (no `Metaspace`
//! support), so this feeds the model what such a pipeline would: whitespace-split
//! words with the SentencePiece delimiter in front. Ids are therefore not a model's
//! real output — the point is the cost of the model stage.
//!
//! One cache capacity per process, one model per process, and an empty cache for every
//! timed repetition. Sharing a warm cache between repetitions times re-encoding a corpus
//! that has already been encoded, which reads as a large win on corpora that in truth
//! have nothing to hit (a Japanese corpus measured 1.6x that way, with 0.2% of its bytes
//! in repeated pre-tokens). To A/B, run it once per capacity:
//!
//!   cargo run --release --example unigram_cache_ab -- data/albert-base-v1-tokenizer.json 0
//!   cargo run --release --example unigram_cache_ab -- data/albert-base-v1-tokenizer.json 65536

use std::time::Instant;

use tk_encode::models::unigram::Unigram;
use tk_encode::pipeline::{Model, PipelineToken};

const REPS: usize = 3;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: unigram_cache_ab <tokenizer.json> <cache capacity> [corpus...]");
    let capacity: usize = args.next().expect("cache capacity").parse().unwrap();

    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let mut model: Unigram = serde_json::from_value(json["model"].clone()).unwrap();
    model.resize_cache(capacity);

    let corpora: Vec<String> = args.collect();
    let corpora = if corpora.is_empty() {
        vec![
            "data/big.txt".to_string(),
            "data/unigram_wagahaiwa_nekodearu.txt".to_string(),
        ]
    } else {
        corpora
    };

    println!("capacity {capacity}");
    for corpus_path in corpora {
        let text = std::fs::read_to_string(&corpus_path).unwrap();
        let corpus: Vec<String> = text
            .split_whitespace()
            .map(|word| format!("\u{2581}{word}"))
            .collect();
        let bytes: usize = corpus.iter().map(String::len).sum();

        // Every rep starts with an empty cache, model-side and scratch-side. Timing reps
        // that share one warm cache measures re-encoding a corpus you have already
        // encoded, which no caller does: a corpus gets one pass, and the only hits it
        // can have are its own repeats.
        let encode = |model: &Unigram| {
            let mut scratch = model.init_scratch();
            let mut output: Vec<PipelineToken> = Vec::with_capacity(1024);
            let mut ids = 0;
            for word in &corpus {
                output.clear();
                model
                    .tokenize_pipeline(word, &mut scratch, &mut output)
                    .unwrap();
                ids += output.len();
            }
            ids
        };

        encode(&model);
        let mut best = f64::INFINITY;
        let mut ids = 0;
        for _ in 0..REPS {
            model.clear_cache();
            let start = Instant::now();
            ids = encode(&model);
            best = best.min(start.elapsed().as_secs_f64());
        }
        println!(
            "  {corpus_path}: {:.1} MB/s ({} pre-tokens, {ids} ids)",
            bytes as f64 / best / 1e6,
            corpus.len(),
        );
    }
}
