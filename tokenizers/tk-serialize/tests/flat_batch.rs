//! Parallel must agree with serial, document for document.
//!
//! A big enough batch goes through the pool and
//! is reassembled from per-worker arenas, while a single document runs straight down the serial
//! loop. Those are the two things that can drift, so that is what this compares.
//!
//! Run `make data/gpt2.json` first -- the fixture is fetched, not committed.

use tk_encode::pipeline::{EncodeOptions, Input, Inputs, Override, PipelineTokenizer};

fn corpus() -> Vec<String> {
    let mut v = Vec::new();
    for i in 0..2000 {
        v.push(format!("the quick brown fox {i} jumps over the lazy dog"));
        v.push(format!("  leading and trailing   {i}  "));
        v.push(format!("语言模型 {i} mixed with ASCII and ελληνικά"));
        v.push(String::new());
        v.push(format!("<|endoftext|> {i} in the middle <|endoftext|>"));
    }
    v
}

fn gpt2() -> PipelineTokenizer {
    let canonical = tk_convert::canonicalize_file("../data/gpt2.json").unwrap();
    tk_serialize::from_json(&canonical).unwrap()
}

/// A batch clears the parallel floor and is reassembled from arenas; one document does not.
fn check(tok: &PipelineTokenizer, add_special: bool) {
    let owned = corpus();
    let options = EncodeOptions {
        add_special_tokens: add_special,
        padding: Override::Off,
    };
    let batched = tok.encode(owned.clone(), &options).wait().unwrap();
    assert_eq!(batched.len(), owned.len(), "document count");

    for (i, text) in owned.iter().enumerate() {
        let alone = tok.encode(text.clone(), &options).wait().unwrap();
        assert_eq!(
            batched[i].ids(),
            alone[0].ids(),
            "document {i} differs for {text:?}"
        );
    }
}

#[test]
fn parallel_matches_serial() {
    let tok = gpt2();
    check(&tok, false);
    check(&tok, true);
}

/// Without specials a pair is just its two sequences, back to back in one document.
#[test]
fn pair_is_both_sequences() {
    let tok = gpt2();
    let options = EncodeOptions {
        add_special_tokens: false,
        padding: Override::Off,
    };
    let (first, second) = ("Hello there".to_string(), "General Kenobi".to_string());

    let a = tok.encode(first.clone(), &options).wait().unwrap();
    let b = tok.encode(second.clone(), &options).wait().unwrap();
    let pair = tok.encode((first, second), &options).wait().unwrap();

    let want: Vec<_> = a[0].ids().iter().chain(b[0].ids()).copied().collect();
    assert_eq!(pair[0].ids(), want, "a pair frames A then B");
}

/// A batch mixing pairs and single sequences keeps each document's own framing.
#[test]
fn pairs_and_singles_in_one_batch() {
    let tok = gpt2();
    let options = EncodeOptions {
        add_special_tokens: false,
        padding: Override::Off,
    };
    let pair = ("Hello there".to_string(), "General Kenobi".to_string());
    let single = "You are a bold one".to_string();

    let batch = tok
        .encode(
            Inputs::Batch(vec![
                Input::Pair(pair.0.clone(), pair.1.clone()),
                Input::Single(single.clone()),
            ]),
            &options,
        )
        .wait()
        .unwrap();

    let alone_pair = tok.encode(pair, &options).wait().unwrap();
    let alone_single = tok.encode(single, &options).wait().unwrap();

    assert_eq!(batch.len(), 2);
    assert_eq!(batch[0].ids(), alone_pair[0].ids());
    assert_eq!(batch[1].ids(), alone_single[0].ids());
}
