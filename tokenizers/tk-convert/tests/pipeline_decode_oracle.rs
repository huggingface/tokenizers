//! The pipeline must decode ids back to the same string as the latest *released* `tokenizers`.
//!
//! The ids come from the release's own `encode_fast`, so decode is judged on decode alone even
//! where encode legitimately diverges.
//!
//!   cargo test -p tk-convert --features bench-baseline --test pipeline_decode_oracle

#![cfg(feature = "bench-baseline")]

mod hub;

use hub::{TEXTS, tokenizer_json};
use tokenizers_release::Tokenizer as Released;

#[test]
fn decoded_text_matches_the_released_crate() {
    let mut diverged = Vec::new();
    for &repo in hub::MODELS {
        let Some(path) = tokenizer_json(repo) else {
            continue;
        };
        let canonical = tk_convert::canonicalize_file(&path)
            .unwrap_or_else(|e| panic!("{repo}: this pass refuses it: {e}"));
        let pipeline = tk_serialize::from_json(&canonical)
            .unwrap_or_else(|e| panic!("{repo}: the canonical reader refuses the conversion: {e}"));
        let released = Released::from_file(&path).expect("the released crate reads it");

        for text in TEXTS {
            let ids = released.encode_fast(*text, true).unwrap().get_ids().to_vec();
            for skip_special in [false, true] {
                let want = released.decode(&ids, skip_special).unwrap();
                let got = pipeline.decode(&ids, skip_special).unwrap_or_default();
                if want != got {
                    diverged.push(format!("{repo} skip={skip_special}: {want:?} vs {got:?}"));
                }
            }
        }
    }
    assert!(diverged.is_empty(), "decoded text diverges:\n{}", diverged.join("\n"));
}
