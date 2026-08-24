//! The pipeline must encode to the same ids as the latest *released* `tokenizers`.
//!
//! Each model is fetched from the Hub, converted by this crate, and read back by the canonical
//! reader -- which is the pairing tk-convert exists to make work. Comparing against the release
//! keeps the oracle independent of anything in this tree.
//!
//!   cargo test -p tk-convert --features bench-baseline --test pipeline_oracle

#![cfg(feature = "bench-baseline")]

mod hub;

use hub::{TEXTS, tokenizer_json};
use tokenizers_release::Tokenizer as Released;

#[test]
fn ids_match_the_released_crate() {
    let mut diverged = Vec::new();
    for &repo in hub::MODELS {
        let Some(path) = tokenizer_json(repo) else {
            continue;
        };
        let canonical = tk_convert::canonicalize_file(&path)
            .unwrap_or_else(|e| panic!("{repo}: this pass refuses it: {e}"));
        // A refusal here is the regression this oracle exists to catch, so it fails, not skips.
        let pipeline = tk_serialize::from_json(&canonical)
            .unwrap_or_else(|e| panic!("{repo}: the canonical reader refuses the conversion: {e}"));
        let released = Released::from_file(&path).expect("the released crate reads it");

        for text in TEXTS {
            for special in [false, true] {
                let want = released
                    .encode_fast(*text, special)
                    .unwrap()
                    .get_ids()
                    .to_vec();
                let got: Vec<u32> = pipeline.encode(*text, special).wait().unwrap()[0]
                    .ids()
                    .iter()
                    .map(|t| t.id())
                    .collect();
                if want != got {
                    diverged.push(format!(
                        "{repo} special={special} {text:?}: {want:?} vs {got:?}"
                    ));
                }
            }
        }
    }
    assert!(diverged.is_empty(), "ids diverge:\n{}", diverged.join("\n"));
}
