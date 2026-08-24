//! The pipeline must decode ids back to the same string as the latest *released* `tokenizers`.
//!
//! The ids come from the release's own `encode_fast`, so decode is judged on decode alone even
//! where encode legitimately diverges. Fixtures are legacy `1.0` files and go through `tk-convert`
//! first.
//!
//! `bert-wiki` is expected to diverge: `PipelineWordPiece` keeps only its forward vocab, so it has
//! no id -> token direction. Left in on purpose, so this stays red until that lands.
//!
//!   cargo test -p tk-convert --features bench-baseline --test pipeline_decode_oracle

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
fn decoded_text_matches_the_released_crate() {
    let mut diverged = Vec::new();
    for &name in MODELS {
        let path = Path::new(DATA).join(name);
        if !path.exists() {
            eprintln!("skip {name}: not present (fetch with `make models`)");
            continue;
        }
        let canonical = tk_convert::canonicalize_file(&path)
            .unwrap_or_else(|e| panic!("{name}: this pass refuses the fixture: {e}"));
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
                if chunk.is_empty() {
                    continue;
                }
                let ids = released.encode_fast(chunk, true).unwrap().get_ids().to_vec();
                for skip_special in [false, true] {
                    let want = released.decode(&ids, skip_special).unwrap();
                    let got = pipeline.decode(&ids, skip_special).unwrap_or_default();
                    if want != got {
                        diverged.push(format!("{name} {group}/{stem} {width}B skip={skip_special}"));
                    }
                }
            }
        }
    }
    assert!(diverged.is_empty(), "decoded text diverges:\n{}", diverged.join("\n"));
}
