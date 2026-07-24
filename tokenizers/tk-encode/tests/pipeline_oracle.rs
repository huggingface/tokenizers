//! `PipelineTokenizer` must encode to exactly the same token ids as the latest
//! *released* `tokenizers` crate — not the in-tree legacy `Tokenizer`, which is
//! being removed (oracles must not depend on it). We compare against the release's
//! `encode_fast` (its offset-free path; the pipeline computes no offsets either),
//! over fixed windows of every fixture corpus, for both `add_special_tokens`
//! values — `true` exercises the post-process frame.
//!
//! One test per model; a failure lists every diverging (fixture, window, flags)
//! case instead of stopping at the first. A model the pipeline can't build or
//! encode yet is skipped, not failed; the model set mirrors the decode oracle's.
//!
//! Behind the `bench-baseline` feature (the released crate is optional):
//!   cargo test -p tk-encode --features bench-baseline --test pipeline_oracle

#![cfg(feature = "bench-baseline")]

mod common;

use std::convert::TryFrom;
use std::path::Path;

use common::{DATA, FIXTURES, WINDOWS, window};
use tk_encode::Tokenizer;
use tk_encode::pipeline::PipelineTokenizer;
use tokenizers_release::Tokenizer as Released;

const PROBE: &str = "The quick brown fox jumps 123.";

fn check_model(tok_file: &str) {
    let path = Path::new(DATA).join(tok_file);
    // The legacy `Tokenizer` only *builds* the pipeline (its sole constructor
    // today); it is never an encode reference. Drops out once a direct loader exists.
    let Ok(tree) = Tokenizer::from_file(&path) else {
        eprintln!("skip {tok_file}: not present (fetch with `make bench-models`)");
        return;
    };
    let Ok(pipeline) = PipelineTokenizer::try_from(&tree) else {
        eprintln!("skip {tok_file}: not supported by PipelineTokenizer");
        return;
    };
    // Build constraints can be met while encode is still unimplemented for this shape.
    if pipeline.encode(PROBE, false).is_err() {
        eprintln!("skip {tok_file}: pipeline can't encode this model yet");
        return;
    }
    let released = match Released::from_file(&path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("skip {tok_file}: released crate can't load it: {e}");
            return;
        }
    };

    let mut failures = Vec::new();
    for &(group, stem) in FIXTURES {
        let fixture = Path::new(DATA)
            .join("fixtures")
            .join(group)
            .join(format!("{stem}.txt"));
        let Ok(text) = std::fs::read_to_string(&fixture) else {
            eprintln!(
                "skip {}: fixture absent (run `make fixtures`)",
                fixture.display()
            );
            continue;
        };
        let mut start = 0;
        for &w in WINDOWS {
            let chunk = window(&text, start, w);
            start += w;
            if chunk.is_empty() {
                continue;
            }
            for add_special_tokens in [false, true] {
                let expected = released
                    .encode_fast(chunk, add_special_tokens)
                    .unwrap()
                    .get_ids()
                    .to_vec();
                let got: Vec<u32> = pipeline
                    .encode(chunk, add_special_tokens)
                    .unwrap()
                    .iter()
                    .map(|t| t.id)
                    .collect();
                if expected != got {
                    let at = expected
                        .iter()
                        .zip(&got)
                        .position(|(e, g)| e != g)
                        .unwrap_or(expected.len().min(got.len()));
                    failures.push(format!(
                        "{group}/{stem} ({w} B window, add_special_tokens={add_special_tokens}): \
                         ids diverge at {at} (expected len {}, got len {})",
                        expected.len(),
                        got.len(),
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{tok_file}: {} case(s) diverge from the released crate:\n{}",
        failures.len(),
        failures.join("\n"),
    );
}

// Same model set as the decode oracle (one per decoder archetype); whichever files
// a given checkout has get run (bert-wiki + llama-3 ship with `make test`; the rest
// with `make bench-models`). Unsupported models skip, not fail.
#[test]
fn bert_wiki() {
    check_model("bert-wiki.json");
}

#[test]
fn bert_base_uncased() {
    check_model("bert-base-uncased.json");
}

#[test]
fn gpt2() {
    check_model("gpt2.json");
}

#[test]
fn llama3() {
    check_model("llama-3-tokenizer.json");
}

#[test]
fn llama2() {
    check_model("llama-2.json");
}

#[test]
fn t5_base() {
    check_model("t5-base.json");
}

#[test]
fn albert() {
    check_model("albert-base-v1-tokenizer.json");
}
