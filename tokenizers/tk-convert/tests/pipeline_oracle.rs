//! The pipeline must encode to the same ids as the latest *released* `tokenizers`, over windows of
//! the real fixture corpora, with and without special tokens.
//!
//! Fixtures are legacy `1.0` files, so they go through `tk-convert` first -- that is the only way a
//! canonical-only reader can load one, and it is the pairing this crate exists to make work.
//!
//!   cargo test -p tk-convert --features bench-baseline --test pipeline_oracle

#![cfg(feature = "bench-baseline")]

mod common;

use common::{DATA, FIXTURES, WINDOWS, window};
use std::path::Path;
use tokenizers_release::Tokenizer as Released;

const MODELS: &[&str] = &[
    "bert-wiki.json",
    "bert-base-uncased.json",
    "gpt2.json",
    "llama-3-tokenizer.json",
    "llama-2.json",
    "gemma-4.json",
    "t5-base.json",
    "albert-base-v1-tokenizer.json",
    "mistral-small-4.json",
];

#[test]
fn ids_match_the_released_crate() {
    let mut diverged = Vec::new();
    for &name in MODELS {
        let path = Path::new(DATA).join(name);
        if !path.exists() {
            eprintln!("skip {name}: not present (fetch with `make models`)");
            continue;
        }
        let canonical = tk_convert::canonicalize_file(&path)
            .unwrap_or_else(|e| panic!("{name}: this pass refuses the fixture: {e}"));
        // A refusal here is the regression this oracle exists to catch, so it fails rather
        // than skips.
        let pipeline = tk_serialize::from_json(&canonical)
            .unwrap_or_else(|e| panic!("{name}: the canonical reader refuses the conversion: {e}"));
        let released = Released::from_file(&path).expect("the released crate reads the fixture");

        for &(group, stem) in FIXTURES {
            let corpus = Path::new(DATA).join("fixtures").join(group).join(format!("{stem}.txt"));
            let Ok(text) = std::fs::read_to_string(&corpus) else {
                continue; // `make fixtures` has not run
            };
            let mut start = 0;
            for &width in WINDOWS {
                let chunk = window(&text, start, width);
                start += width;
                for special in [false, true] {
                    if chunk.is_empty() {
                        continue;
                    }
                    let want = released.encode_fast(chunk, special).unwrap().get_ids().to_vec();
                    let got: Vec<u32> = pipeline.encode(chunk, special).wait().unwrap()[0]
                        .ids()
                        .iter()
                        .map(|t| t.id())
                        .collect();
                    if want != got {
                        diverged.push(format!("{name} {group}/{stem} {width}B special={special}"));
                    }
                }
            }
        }
    }
    assert!(diverged.is_empty(), "ids diverge:\n{}", diverged.join("\n"));
}
