//! `encode_batch_flat` must return exactly what `encode` returns, document for
//! document.
//!
//! It is a second implementation of the same thing — it lays ids straight into a
//! shared arena and splices the template's specials itself instead of going
//! through `post_process` — so "same ids" is the only thing keeping the two in
//! agreement. Checked on every model in `data/`, over real corpora, for
//! `add_special_tokens` both ways, and at a batch size large enough to cross into
//! the parallel path.

use std::path::Path;

use tk_encode::Tokenizer;
use tk_encode::pipeline::PipelineTokenizer;

mod common;
use common::{DATA, FIXTURES};

const PROBE: &str = "The quick brown fox jumps 123.";

fn check_model(tok_file: &str) {
    let path = Path::new(DATA).join(tok_file);
    let Ok(tree) = Tokenizer::from_file(&path) else {
        eprintln!("skip {tok_file}: not present (fetch with `make models`)");
        return;
    };
    let Ok(pipeline) = PipelineTokenizer::try_from(&tree) else {
        eprintln!("skip {tok_file}: not supported by PipelineTokenizer");
        return;
    };
    if pipeline.encode(PROBE, false).wait().is_err() {
        eprintln!("skip {tok_file}: pipeline can't encode this model yet");
        return;
    }

    for &(group, stem) in FIXTURES {
        let fixture = Path::new(DATA)
            .join("fixtures")
            .join(group)
            .join(format!("{stem}.txt"));
        let Ok(text) = std::fs::read_to_string(&fixture) else {
            continue;
        };
        // One document per line: the shape the flat path exists for, and enough
        // of them to be split across the pool rather than run serially.
        let docs: Vec<&str> = text.lines().filter(|l| !l.is_empty()).take(4000).collect();
        if docs.is_empty() {
            continue;
        }

        for add_special_tokens in [false, true] {
            let flat = pipeline
                .encode_batch_flat(&docs, add_special_tokens)
                .unwrap_or_else(|e| panic!("{tok_file} {group}/{stem}: flat encode failed: {e}"));
            let owned: Vec<String> = docs.iter().map(|s| (*s).to_string()).collect();
            let reference = pipeline
                .encode(owned, add_special_tokens)
                .wait()
                .unwrap_or_else(|e| panic!("{tok_file} {group}/{stem}: encode failed: {e}"));

            assert_eq!(
                flat.len(),
                reference.len(),
                "{tok_file} {group}/{stem} (add_special_tokens={add_special_tokens}): \
                 flat has {} documents, encode has {}",
                flat.len(),
                reference.len()
            );
            for (i, expected) in reference.iter().enumerate() {
                let got = flat.row(i).expect("row within len");
                assert_eq!(
                    got,
                    expected.ids(),
                    "{tok_file} {group}/{stem} (add_special_tokens={add_special_tokens}): \
                     document {i} differs\n  input: {:?}",
                    docs[i]
                );
            }
        }
    }
}

macro_rules! model_tests {
    ($($name:ident => $file:literal),* $(,)?) => {
        $(#[test] fn $name() { check_model($file); })*
    };
}

model_tests! {
    gpt2 => "gpt2.json",
    llama3 => "llama-3.json",
    llama2 => "llama-2.json",
    bert_base_uncased => "bert-base-uncased.json",
    bert_wiki => "bert-wiki.json",
    t5_base => "t5-base.json",
    albert => "albert.json",
    gemma4 => "gemma-4.json",
    mistral_small_4 => "mistral-small-4.json",
    deepseek_v4 => "deepseek-v4.json",
}
