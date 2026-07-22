//! `PipelineTokenizer` must decode ids back to exactly the same string as the
//! latest *released* `tokenizers` crate — not the in-tree legacy `Tokenizer`,
//! which is being removed (oracles must not depend on it).
//!
//! Round-trip: encode a fixture window with the release's `encode_fast`
//! (`add_special_tokens = false`), then decode those ids with both the released
//! `decode` and `PipelineTokenizer::decode` and require byte-identical output.
//! Feeding the release's own ids keeps this honest even where the pipeline's
//! *encode* legitimately diverges — decode is judged on decode alone. (0.23.1 has
//! no `decode_fast`; switch to it if a future release adds one.)
//!
//! Inputs and covered tokenizers mirror the encode oracle (`pipeline_oracle.rs`):
//! bert-wiki (WordPiece `##`) and llama-3 (byte-level) over seeded-random windows
//! of the fixture corpora.
//!
//! Behind `bench-baseline`. IGNORED until `PipelineTokenizer::decode` is
//! implemented (a loud stub today, so these fail on `.unwrap()`); un-ignore with
//! the impl by dropping the `#[ignore]` lines:
//!   cargo test -p tk-encode --features bench-baseline --test pipeline_decode_oracle -- --ignored

#![cfg(feature = "bench-baseline")]

mod common;

use std::convert::TryFrom;
use std::path::Path;

use common::{DATA, WINDOWS, fixture_files, random_chunk};
use tk_encode::Tokenizer;
use tk_encode::pipeline::PipelineTokenizer;
use tokenizers_release::Tokenizer as Released;

fn check(tok_file: &str) {
    let path = Path::new(DATA).join(tok_file);
    // The legacy `Tokenizer` only *builds* the pipeline (its sole constructor
    // today); it is never a decode reference. Drops out once a direct loader exists.
    let Ok(tree) = Tokenizer::from_file(&path) else {
        eprintln!("skip {tok_file}: not present (fetch with `make fixtures bench-models`)");
        return;
    };
    let Ok(pipeline) = PipelineTokenizer::try_from(&tree) else {
        eprintln!("skip {tok_file}: not supported by PipelineTokenizer");
        return;
    };
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
            let ids = released
                .encode_fast(chunk, false)
                .unwrap()
                .get_ids()
                .to_vec();
            let expected = released.decode(&ids, false).unwrap();
            let got = pipeline.decode(&ids, false).unwrap();
            assert_eq!(
                expected,
                got,
                "decode mismatch on {} @ {:?}",
                f.display(),
                chunk.chars().take(60).collect::<String>(),
            );
        }
    }
}

macro_rules! decode_tests {
    ($($name:ident => $tok:literal),* $(,)?) => {
        $(
            #[test]
            #[ignore = "un-ignore once PipelineTokenizer::decode is implemented"]
            fn $name() {
                check($tok);
            }
        )*
    };
}

// Mirrors the encode oracle's model set; missing files are skipped.
decode_tests! {
    bert_wiki         => "bert-wiki.json",
    bert_base_uncased => "bert-base-uncased.json",
    gpt2              => "gpt2.json",
    llama3            => "llama-3-tokenizer.json",
}
