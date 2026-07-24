//! `PipelineTokenizer` must encode to exactly the same token ids as the latest
//! *released* `tokenizers` crate — not the in-tree legacy `Tokenizer`, which is
//! being removed (oracles must not depend on it). We compare against the release's
//! `encode_fast` (its offset-free path; the pipeline computes no offsets either),
//! over seeded-random windows of the fixture corpora.
//!
//! One test per `model::fixture` — so a red run names the exact model and
//! language/modality that broke (same structure as `pipeline_decode_oracle.rs`; no
//! `{single,pair}` split here because `PipelineTokenizer::encode` takes a single
//! sequence). Every window is checked for both `add_special_tokens` values —
//! `true` exercises the post-process frame. A model the pipeline can't build or
//! encode yet is skipped, not failed; the model set mirrors the decode oracle's.
//!
//! Behind the `bench-baseline` feature (the released crate is optional):
//!   cargo test -p tk-encode --features bench-baseline --test pipeline_oracle

#![cfg(feature = "bench-baseline")]

mod common;

use std::convert::TryFrom;
use std::path::Path;

use common::{DATA, WINDOWS, random_chunk, stem_seed};
use tk_encode::Tokenizer;
use tk_encode::pipeline::PipelineTokenizer;
use tokenizers_release::Tokenizer as Released;

const PROBE: &str = "The quick brown fox jumps 123.";

/// One (model, fixture) case: encode every window of `data/fixtures/{group}/
/// {stem}.txt` with both sides and require identical ids.
fn check_one(tok_file: &str, group: &str, stem: &str) {
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

    let fixture = Path::new(DATA)
        .join("fixtures")
        .join(group)
        .join(format!("{stem}.txt"));
    let Ok(text) = std::fs::read_to_string(&fixture) else {
        eprintln!(
            "skip {}: fixture absent (run `make fixtures`)",
            fixture.display()
        );
        return;
    };
    if text.is_empty() {
        return;
    }

    for (w, &window) in WINDOWS.iter().enumerate() {
        let chunk = random_chunk(&text, window, stem_seed(stem) ^ w as u64);
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
            assert_eq!(
                expected,
                got,
                "id mismatch on {tok_file} [{group}/{stem}] \
                 (add_special_tokens={add_special_tokens}) @ {:?}",
                chunk.chars().take(60).collect::<String>(),
            );
        }
    }
}

// Same model set as the decode oracle (one per decoder archetype); whichever files
// a given checkout has get run (bert-wiki + llama-3 ship with `make test`; the rest
// with `make bench-models`). Unsupported models skip, not fail.
macro_rules! encode_tests {
    ($($model:ident => $tok:literal),* $(,)?) => {
        $(
            // Fixture-derived fn names (amh_Ethi, tam_Taml, …) keep their script
            // casing so a failure reads as the fixture, not a mangled snake_case.
            #[allow(non_snake_case)]
            mod $model {
                crate::common::for_each_fixture!(crate::check_one, $tok);
            }
        )*
    };
}

encode_tests! {
    bert_wiki         => "bert-wiki.json",
    bert_base_uncased => "bert-base-uncased.json",
    gpt2              => "gpt2.json",
    llama3            => "llama-3-tokenizer.json",
    llama2            => "llama-2.json",
    t5_base           => "t5-base.json",
    albert            => "albert-base-v1-tokenizer.json",
}
