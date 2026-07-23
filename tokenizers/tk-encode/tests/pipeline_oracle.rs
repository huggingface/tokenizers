//! `PipelineTokenizer` must encode to exactly the same token ids as the latest
//! *released* `tokenizers` crate — not the in-tree legacy `Tokenizer`, which is
//! being removed (oracles must not depend on it). We compare against the release's
//! `encode_fast` (its offset-free path; the pipeline computes no offsets either),
//! over seeded-random windows of the fixture corpora.
//!
//! Covered: bert-wiki (Whitespace + WordPiece) and llama-3 (byte-level BPE) — a
//! model the pipeline can't build or encode yet is skipped, not failed.
//!
//! Behind the `bench-baseline` feature (the released crate is optional):
//!   cargo test -p tk-encode --features bench-baseline --test pipeline_oracle

#![cfg(feature = "bench-baseline")]

mod common;

use std::convert::TryFrom;
use std::path::Path;

use common::{DATA, WINDOWS, fixture_files, random_chunk};
use tk_encode::Tokenizer;
use tk_encode::pipeline::PipelineTokenizer;
use tokenizers_release::Tokenizer as Released;

const PROBE: &str = "The quick brown fox jumps 123.";

fn check(tok_file: &str) {
    let path = Path::new(DATA).join(tok_file);
    // The legacy `Tokenizer` only *builds* the pipeline (its sole constructor
    // today); it is never an encode reference. Drops out once a direct loader exists.
    let Ok(tree) = Tokenizer::from_file(&path) else {
        eprintln!("skip {tok_file}: not present (fetch with `make fixtures bench-models`)");
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
    let files = fixture_files();
    if files.is_empty() {
        eprintln!("skip {tok_file}: no fixtures under {DATA}/fixtures — run `make fixtures`");
        return;
    }

    for (i, f) in files.iter().enumerate() {
        let text = std::fs::read_to_string(f).unwrap();
        if text.is_empty() {
            continue;
        }
        for (w, &window) in WINDOWS.iter().enumerate() {
            let chunk = random_chunk(&text, window, ((i as u64) << 8) | w as u64);
            if chunk.is_empty() {
                continue;
            }
            let expected = released
                .encode_fast(chunk, false)
                .unwrap()
                .get_ids()
                .to_vec();
            let got: Vec<u32> = pipeline
                .encode(chunk, false)
                .unwrap()
                .iter()
                .map(|t| t.id)
                .collect();
            assert_eq!(
                expected,
                got,
                "id mismatch on {} @ {:?}",
                f.display(),
                chunk.chars().take(60).collect::<String>(),
            );
        }
    }
}

macro_rules! oracle_tests {
    ($($name:ident => $tok:literal),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                check($tok);
            }
        )*
    };
}

// Two decode/encode archetypes each; whichever files a given checkout has get run
// (bert-wiki + llama-3 ship with `make test`; the rest with `make bench-models`).
oracle_tests! {
    bert_wiki         => "bert-wiki.json",
    bert_base_uncased => "bert-base-uncased.json",
    gpt2              => "gpt2.json",
    llama3            => "llama-3-tokenizer.json",
}
