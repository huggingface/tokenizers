//! `PipelineTokenizer` must decode ids back to exactly the same string as the
//! latest *released* `tokenizers` crate — not the in-tree legacy `Tokenizer`,
//! which is being removed (oracles must not depend on it).
//!
//! Round-trip: encode a fixture window with the release's `encode_fast`, then
//! decode those ids with both the released `decode` and
//! `PipelineTokenizer::decode` and require byte-identical output. Feeding the
//! release's own ids keeps this honest even where the pipeline's *encode*
//! legitimately diverges — decode is judged on decode alone. (0.23.1 has no
//! `decode_fast`; switch to it if a future release adds one.)
//!
//! One test per `model::{single,pair}::fixture` — so a red run names the exact
//! model, input kind, and language/modality that broke, instead of one giant
//! per-model test that stops at its first mismatch. Inside each, the window ×
//! `add_special_tokens` × `skip_special_tokens` sweep runs, comparing the pipeline
//! to the release three ways: one-shot `decode`, chunk-by-chunk `decode_stream`,
//! and `decode_batch` over the fixture's windows. Together they exercise:
//!   - the per-model **decoder** — one model per `DecoderWrapper` a fixture reaches
//!     (see the `decode_tests!` set below);
//!   - **special tokens** in the stream — `add_special_tokens = true` injects the
//!     model's specials ([CLS]/[SEP], <|begin_of_text|>, …); `pair` adds the pair
//!     template's extras (second [SEP], type ids). Single-sequence models have no
//!     pair template, so their `pair` tests skip;
//!   - **`skip_special_tokens`** — both keeping and dropping those ids;
//!   - the multibyte paths (byte-fallback `<0xNN>`, Metaspace) that only fire on the
//!     non-ASCII language fixtures — run `make fixtures` or every fixture skips.
//!
//! `non_special_added_token_survives_skip` covers the one case the corpus can't: a
//! *non-special* added token (every fixture's added tokens are `special: true`) must
//! survive `skip_special_tokens = true`. It's built by adding the same token to both
//! sides, so it stays an honest release-vs-pipeline parity check.
//!
//! Behind `bench-baseline`. These FAIL until `PipelineTokenizer::decode` applies the
//! decoder and distinguishes special from non-special added vocab — on purpose: CI
//! stays red until decode lands, rather than hiding the gap behind a skipped test.
//!   cargo test -p tk-encode --features bench-baseline --test pipeline_decode_oracle

#![cfg(feature = "bench-baseline")]

mod common;

use std::convert::TryFrom;
use std::path::Path;

use common::{DATA, WINDOWS, random_chunk, stem_seed};
use tk_encode::Tokenizer;
use tk_encode::pipeline::PipelineTokenizer;
use tokenizers_release::Tokenizer as Released;

/// One (model, input-kind, fixture) case: decode every window of `data/fixtures/
/// {group}/{stem}.txt` and require pipeline == release for one-shot, streaming, and
/// batch decode across the `add_special_tokens × skip_special_tokens` sweep.
fn check_one(tok_file: &str, pair: bool, group: &str, stem: &str) {
    let path = Path::new(DATA).join(tok_file);
    // The legacy `Tokenizer` only *builds* the pipeline (its sole constructor
    // today); it is never a decode reference. Drops out once a direct loader exists.
    let Ok(tree) = Tokenizer::from_file(&path) else {
        eprintln!("skip {tok_file}: not present (fetch with `make bench-models`)");
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
    // Single-sequence tokenizers have no pair template and error on a pair input.
    if pair && released.encode_fast(("a", "b"), false).is_err() {
        eprintln!("skip {tok_file} [{group}/{stem}] pair: model has no pair template");
        return;
    }

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

    let mut batch: Vec<Vec<u32>> = Vec::new();
    for (w, &window) in WINDOWS.iter().enumerate() {
        let seed = stem_seed(stem) ^ w as u64;
        let chunk = random_chunk(&text, window, seed);
        if chunk.is_empty() {
            continue;
        }
        for add_special_tokens in [false, true] {
            let ids = if pair {
                // A second, disjoint-seeded slice so the pair drives the
                // post-processor's pair template into the decoded id stream.
                let second = random_chunk(&text, window, seed ^ 0xA5A5_A5A5);
                match released.encode_fast((chunk, second), add_special_tokens) {
                    Ok(enc) => enc.get_ids().to_vec(),
                    Err(_) => continue,
                }
            } else {
                released
                    .encode_fast(chunk, add_special_tokens)
                    .unwrap()
                    .get_ids()
                    .to_vec()
            };

            for skip_special_tokens in [false, true] {
                let expected = released.decode(&ids, skip_special_tokens).unwrap();
                let one_shot = pipeline.decode(&ids, skip_special_tokens).unwrap();
                let streamed = stream_decode(&pipeline, &ids, skip_special_tokens);
                let ctx = format!(
                    "{tok_file} [{group}/{stem}] (pair={pair}, \
                     add_special_tokens={add_special_tokens}, \
                     skip_special_tokens={skip_special_tokens}) @ {:?}",
                    chunk.chars().take(60).collect::<String>(),
                );
                assert_eq!(expected, one_shot, "decode mismatch on {ctx}");
                assert_eq!(expected, streamed, "decode_stream mismatch on {ctx}");
            }
            batch.push(ids);
        }
    }

    // decode_batch must match the released batch decoder over this fixture's ids.
    if !batch.is_empty() {
        let sentences: Vec<&[u32]> = batch.iter().map(Vec::as_slice).collect();
        assert_eq!(
            released.decode_batch(&sentences, false).unwrap(),
            pipeline.decode_batch(&sentences, false).unwrap(),
            "decode_batch mismatch for {tok_file} [{group}/{stem}] (pair={pair})",
        );
    }
}

/// Feed `ids` through [`PipelineTokenizer::decode_stream`] one at a time and
/// concatenate the emitted chunks — for a complete id sequence this must equal a
/// one-shot `decode`, so the oracle can compare it against the release directly.
fn stream_decode(pipeline: &PipelineTokenizer, ids: &[u32], skip_special_tokens: bool) -> String {
    let mut stream = pipeline.decode_stream(skip_special_tokens);
    let mut out = String::new();
    for &id in ids {
        if let Some(chunk) = stream.step(id).unwrap() {
            out.push_str(&chunk);
        }
    }
    out
}

/// A non-special *added* token must survive `skip_special_tokens = true` — that
/// flag drops only tokens marked `special`. Every fixture's added tokens are
/// `special: true`, so this can't come from the corpus: add the same non-special
/// token to both the reference and the pipeline's source, then round-trip a string
/// containing it. Byte-level gpt2 decodes this ASCII round-trip correctly, so the
/// only thing under test here is the special-vs-non-special distinction.
#[test]
fn non_special_added_token_survives_skip() {
    let path = Path::new(DATA).join("gpt2.json");
    let Ok(mut tree) = Tokenizer::from_file(&path) else {
        eprintln!("skip: gpt2.json not present (fetch with `make bench-models`)");
        return;
    };
    let Ok(mut released) = Released::from_file(&path) else {
        eprintln!("skip: released crate can't load gpt2.json");
        return;
    };

    let content = "<|mytok|>";
    tree.add_tokens([tk_encode::AddedToken::from(content, false)])
        .unwrap();
    released
        .add_tokens([tokenizers_release::AddedToken::from(content, false)])
        .unwrap();
    let pipeline = PipelineTokenizer::try_from(&tree).unwrap();

    let text = format!("hello {content} world");
    let ids = released
        .encode_fast(text.as_str(), false)
        .unwrap()
        .get_ids()
        .to_vec();
    let tok_id = released.token_to_id(content).unwrap();
    assert!(
        ids.contains(&tok_id),
        "added token not present in id stream"
    );

    for skip_special_tokens in [false, true] {
        let expected = released.decode(&ids, skip_special_tokens).unwrap();
        let got = pipeline.decode(&ids, skip_special_tokens).unwrap();
        assert_eq!(expected, got, "skip_special_tokens={skip_special_tokens}");
        assert!(
            got.contains(content),
            "non-special added token dropped at skip_special_tokens={skip_special_tokens}: {got:?}",
        );
    }
}

// One model per distinct decoder so every `DecoderWrapper` variant a fixture can
// reach is exercised: byte-level (gpt2, llama-3), WordPiece (bert-base-uncased),
// null→space-join (bert-wiki), Metaspace (t5-base, albert), and the SentencePiece
// `Sequence[Replace, ByteFallback, Fuse, Strip]` (llama-2). Missing files or
// unsupported models are skipped, not ignored: the decode gaps fail on purpose
// until decode is implemented. (BPE/CTC decoders have no fixture in this set yet.)
macro_rules! decode_tests {
    ($($model:ident => $tok:literal),* $(,)?) => {
        $(
            mod $model {
                // Fixture-derived fn names (amh_Ethi, tam_Taml, …) keep their script
                // casing so a failure reads as the fixture, not a mangled snake_case.
                #[allow(non_snake_case)]
                mod single {
                    crate::common::for_each_fixture!(crate::check_one, $tok, false);
                }
                #[allow(non_snake_case)]
                mod pair {
                    crate::common::for_each_fixture!(crate::check_one, $tok, true);
                }
            }
        )*
    };
}

decode_tests! {
    bert_wiki         => "bert-wiki.json",
    bert_base_uncased => "bert-base-uncased.json",
    gpt2              => "gpt2.json",
    llama3            => "llama-3-tokenizer.json",
    llama2            => "llama-2.json",
    t5_base           => "t5-base.json",
    albert            => "albert-base-v1-tokenizer.json",
}
