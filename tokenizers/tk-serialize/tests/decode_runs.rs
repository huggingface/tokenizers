//! A control token and its spelling must not decode to the same thing.
//!
//! A response parser matches its delimiters in decoded text, so a model that quotes
//! `<|endoftext|>` in an answer looks exactly like one that ended its turn, and the answer is cut
//! short. `decode_runs` keeps the two apart: the delimiter is an id, and the spelling is text.

use tk_encode::pipeline::Decoded;
use tk_encode::pipeline::EncodeOptions;
use tk_encode::pipeline::Tokenizer;

/// `data/gpt2.json` is still version `1.0`, so run the upgrade pass first -- the reader only
/// accepts canonical `2.0`. Same as `benches/encode.rs`.
fn load() -> Tokenizer {
    let canonical = tk_convert::canonicalize_file("../data/gpt2.json").unwrap();
    tk_serialize::from_json(&canonical).unwrap()
}

const EOT: u32 = 50256;
const MARKER: &str = "<|endoftext|>";

fn encode(tokenizer: &Tokenizer, text: &str, encode_special_tokens: bool) -> Vec<u32> {
    let mut options = EncodeOptions::default();
    options.encode_special_tokens = encode_special_tokens;
    tokenizer
        .encode(text, &options)
        .wait()
        .unwrap()
        .into_iter()
        .flat_map(|encoding| encoding.ids().iter().map(|token| token.id()).collect::<Vec<u32>>())
        .collect()
}

#[test]
fn a_control_token_is_an_id_and_its_spelling_is_text() {
    let tokenizer = load();

    let as_token = encode(&tokenizer, MARKER, false);
    assert_eq!(as_token, [EOT]);
    assert_eq!(
        tokenizer.decode_runs(&as_token, false).unwrap(),
        [Decoded::Added {
            id: EOT,
            text: MARKER.to_string()
        }]
    );

    // `encode_special_tokens` sends the marker through the model, which is what a model quoting it
    // in prose produces: the same characters, and not the token.
    let spelled_out = encode(&tokenizer, MARKER, true);
    assert!(!spelled_out.contains(&EOT));
    let runs = tokenizer.decode_runs(&spelled_out, false).unwrap();
    assert_eq!(runs, [Decoded::Text(MARKER.to_string())]);
}

#[test]
fn the_runs_concatenate_to_the_plain_decode() {
    let tokenizer = load();
    let text = "a<|endoftext|>b";
    let ids = encode(&tokenizer, text, false);
    assert!(ids.contains(&EOT));

    let runs = tokenizer.decode_runs(&ids, false).unwrap();
    assert_eq!(runs.len(), 3, "text, the token, text");
    assert!(matches!(&runs[1], Decoded::Added { id, .. } if *id == EOT));

    let joined: String = runs
        .iter()
        .map(|run| match run {
            Decoded::Text(text) => text.as_str(),
            Decoded::Added { text, .. } => text.as_str(),
        })
        .collect();
    assert_eq!(joined, tokenizer.decode(&ids, false).unwrap());
    assert_eq!(joined, text);
}

#[test]
fn skipping_specials_drops_the_added_run() {
    let tokenizer = load();
    let ids = encode(&tokenizer, "a<|endoftext|>b", false);

    let runs = tokenizer.decode_runs(&ids, true).unwrap();
    assert!(!runs.iter().any(|run| matches!(run, Decoded::Added { .. })));
    assert_eq!(tokenizer.decode(&ids, true).unwrap(), "ab");
}

#[test]
fn two_delimiters_are_two_runs() {
    let tokenizer = load();
    let text = format!("{MARKER}{MARKER}");
    let ids = encode(&tokenizer, &text, false);
    assert_eq!(ids, [EOT, EOT]);

    let runs = tokenizer.decode_runs(&ids, false).unwrap();
    assert_eq!(runs.len(), 2, "not one span of two delimiters");
    assert!(runs.iter().all(|run| matches!(run, Decoded::Added { id, .. } if *id == EOT)));
}
