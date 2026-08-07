//! `PipelineTokenizer` must decode ids back to exactly the same string as the
//! latest *released* `tokenizers` crate — not the in-tree legacy `Tokenizer`,
//! which is being removed (oracles must not depend on it).
//!
//! Round-trip: encode fixture windows with the release's `encode_fast`, then
//! decode those ids with both the released `decode` and
//! `PipelineTokenizer::decode` and require byte-identical output. Feeding the
//! release's own ids keeps this honest even where the pipeline's *encode*
//! legitimately diverges — decode is judged on decode alone. (0.23.1 has no
//! `decode_fast`; switch to it if a future release adds one.)
//!
//! One test per model; a failure lists every diverging (fixture, window, flags)
//! case instead of stopping at the first. Each case sweeps single and pair inputs
//! (pair drives the pair template's extras — second [SEP], type ids; models
//! without a pair template only run single), `add_special_tokens` ×
//! `skip_special_tokens`, and compares one-shot `decode`, chunk-by-chunk
//! `decode_stream`, and `decode_batch`. The multibyte paths (byte-fallback
//! `<0xNN>`, Metaspace) only fire on the non-ASCII language fixtures — run
//! `make fixtures` or every fixture skips.
//!
//! `non_special_added_token_survives_skip` covers the one case the corpus can't: a
//! *non-special* added token (every fixture's added tokens are `special: true`) must
//! survive `skip_special_tokens = true`. It's built by adding the same token to both
//! sides, so it stays an honest release-vs-pipeline parity check.
//!
//! Behind `bench-baseline`. `bert_wiki` still FAILS: `PipelineWordPiece` keeps only its
//! forward `vocab_trie`, so it has no id → token direction to decode with — on purpose: CI
//! stays red until that lands, rather than hiding the gap behind a skipped test.
//!   cargo test -p tk-encode --features bench-baseline --test pipeline_decode_oracle

#![cfg(feature = "bench-baseline")]

mod common;

use std::convert::TryFrom;
use std::path::Path;

use common::{DATA, FIXTURES, WINDOWS, window};
use tk_encode::Tokenizer;
use tk_encode::pipeline::PipelineTokenizer;
use tokenizers_release::Tokenizer as Released;

fn check_model(tok_file: &str) {
    let path = Path::new(DATA).join(tok_file);
    // The legacy `Tokenizer` only *builds* the pipeline (its sole constructor
    // today); it is never a decode reference. Drops out once a direct loader exists.
    let Ok(tree) = Tokenizer::from_file(&path) else {
        eprintln!("skip {tok_file}: not present (fetch with `make models`)");
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
    let pair_kinds: &[bool] = if released.encode_fast(("a", "b"), false).is_ok() {
        &[false, true]
    } else {
        &[false]
    };

    let mut failures = Vec::new();
    let mut batch: Vec<Vec<u32>> = Vec::new();
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
            // The pair's second sequence: the bytes right after `chunk`.
            let second = window(&text, start + w, w);
            start += w;
            if chunk.is_empty() {
                continue;
            }
            for &pair in pair_kinds {
                for add_special_tokens in [false, true] {
                    let ids = if pair {
                        released
                            .encode_fast((chunk, second), add_special_tokens)
                            .unwrap()
                            .get_ids()
                            .to_vec()
                    } else {
                        released
                            .encode_fast(chunk, add_special_tokens)
                            .unwrap()
                            .get_ids()
                            .to_vec()
                    };
                    for skip_special_tokens in [false, true] {
                        let expected = released.decode(&ids, skip_special_tokens).unwrap();
                        let ctx = format!(
                            "{group}/{stem} ({w} B window, pair={pair}, \
                             add_special_tokens={add_special_tokens}, \
                             skip_special_tokens={skip_special_tokens})"
                        );
                        match pipeline.decode(&ids, skip_special_tokens) {
                            Ok(got) if got == expected => {}
                            Ok(_) => failures.push(format!("{ctx}: decode mismatch")),
                            Err(e) => failures.push(format!("{ctx}: decode error: {e}")),
                        }
                        match stream_decode(&pipeline, &ids, skip_special_tokens) {
                            Ok(got) if got == expected => {}
                            Ok(_) => failures.push(format!("{ctx}: decode_stream mismatch")),
                            Err(e) => failures.push(format!("{ctx}: decode_stream error: {e}")),
                        }
                    }
                    batch.push(ids);
                }
            }
        }
    }

    // decode_batch must match the released batch decoder over the whole id set.
    if !batch.is_empty() {
        let sentences: Vec<&[u32]> = batch.iter().map(Vec::as_slice).collect();
        let expected = released.decode_batch(&sentences, false).unwrap();
        match pipeline.decode_batch(&sentences, false) {
            Ok(got) if got == expected => {}
            Ok(_) => failures.push("decode_batch mismatch".into()),
            Err(e) => failures.push(format!("decode_batch error: {e}")),
        }
    }

    let shown: Vec<&str> = failures.iter().take(10).map(String::as_str).collect();
    assert!(
        failures.is_empty(),
        "{tok_file}: {} case(s) diverge from the released crate (first {}):\n{}",
        failures.len(),
        shown.len(),
        shown.join("\n"),
    );
}

/// Feed `ids` through [`PipelineTokenizer::decode_stream`] one at a time and
/// concatenate the emitted chunks — for a complete id sequence this must equal a
/// one-shot `decode`, so the oracle can compare it against the release directly.
fn stream_decode(
    pipeline: &PipelineTokenizer,
    ids: &[u32],
    skip_special_tokens: bool,
) -> tk_encode::Result<String> {
    let mut stream = pipeline.decode_stream(skip_special_tokens);
    let mut out = String::new();
    for &id in ids {
        if let Some(chunk) = stream.step(id)? {
            out.push_str(&chunk);
        }
    }
    Ok(out)
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
        eprintln!("skip: gpt2.json not present (fetch with `make models`)");
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
// unsupported models skip, not fail: the decode gaps fail on purpose until decode
// is implemented. (BPE/CTC decoders have no fixture in this set yet.)
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
