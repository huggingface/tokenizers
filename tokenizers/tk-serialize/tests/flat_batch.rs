//! `encode_batch_flat` has to agree with `encode`, on every template shape.
//!
//! Three shapes matter, and the middle one is why `flat_template` exists:
//!
//! * a template that reproduces the sequence (byte-level BPE, gpt2) -- the flat path writes the
//!   model's output straight into its arena;
//! * a template that wraps the sequence in specials (bert, deepseek with `add_special_tokens`) --
//!   the flat path extends by the prefix and suffix slices;
//! * a template a flat layout cannot express -- the flat path must fall back and still agree.
//!
//! Also run past `PARALLEL_MIN_BYTES` with several documents, so the parallel arena path is the one
//! under test rather than the serial loop.

use tk_encode::pipeline::PipelineTokenizer;

fn load(name: &str) -> PipelineTokenizer {
    let canonical =
        tk_convert::canonicalize_file(&format!("../data/{name}")).expect("canonicalize");
    tk_serialize::from_json(&canonical).expect("read")
}

/// Enough documents, and enough bytes, to clear the parallel threshold.
fn docs() -> Vec<String> {
    (0..400)
        .map(|i| {
            format!("The quick brown fox {i} jumps over the lazy dog, repeatedly and at length.")
        })
        .collect()
}

fn check(name: &str, add_special_tokens: bool) {
    let pipe = load(name);
    let owned = docs();
    let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
    assert!(
        refs.iter().map(|s| s.len()).sum::<usize>() > 8 * 1024,
        "corpus must clear PARALLEL_MIN_BYTES to exercise the parallel path"
    );

    let general = pipe
        .encode(owned.clone(), add_special_tokens)
        .wait()
        .expect("encode");
    let flat = pipe
        .encode_batch_flat(&refs, add_special_tokens)
        .expect("encode_batch_flat");

    assert_eq!(flat.len(), general.len(), "{name}: document count");
    for (i, enc) in general.iter().enumerate() {
        assert_eq!(
            flat.row(i).expect("row in range"),
            enc.ids(),
            "{name}: document {i} (add_special_tokens={add_special_tokens})"
        );
    }
    // The offsets have to describe the ids they came with.
    assert_eq!(
        *flat.offsets().last().expect("offsets non-empty") as usize,
        flat.ids().len()
    );
}

#[test]
fn flat_matches_general_for_a_byte_level_template() {
    check("gpt2.json", true);
    check("gpt2.json", false);
}

#[test]
fn flat_matches_general_when_the_template_wraps_in_specials() {
    // bert's template is `[CLS] A [SEP]`, so this is the prefix/suffix path.
    check("bert-base-uncased.json", true);
    check("bert-base-uncased.json", false);
}

#[test]
fn flat_matches_general_for_a_deepseek_template() {
    check("deepseek-v4.json", true);
    check("deepseek-v4.json", false);
}

#[test]
fn an_empty_batch_is_empty() {
    let pipe = load("gpt2.json");
    let flat = pipe.encode_batch_flat(&[], true).expect("flat");
    assert!(flat.is_empty());
    assert_eq!(flat.offsets(), &[0]);
    assert!(flat.row(0).is_none());
}
