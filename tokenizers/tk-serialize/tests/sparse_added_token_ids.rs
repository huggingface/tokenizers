#![cfg(all(
    feature = "wordlevel",
    feature = "wordpiece",
    feature = "unigram",
    feature = "serialize"
))]

use serde_json::{Value, json};
use std::collections::HashSet;
use tk_encode::pipeline::{EncodeOptions, PipelineTokenizer};

fn model(kind: &str, id: u32) -> Value {
    match kind {
        "BPE" => json!({
            "type": "BPE", "byte_level": false,
            "vocab": {"<unk>": 0, "a": id}, "merges": []
        }),
        "WordPiece" => json!({
            "type": "WordPiece", "unk_token": "<unk>",
            "continuing_subword_prefix": "##", "max_input_chars_per_word": 100,
            "vocab": {"<unk>": 0, "a": id}
        }),
        "WordLevel" => json!({
            "type": "WordLevel", "unk_token": "<unk>",
            "vocab": {"<unk>": 0, "a": id}
        }),
        _ => panic!("unexpected model"),
    }
}

fn config(model: Value, tokens: &[&str]) -> String {
    let added: Vec<_> = tokens
        .iter()
        .enumerate()
        .map(|(id, content)| {
            json!({
                "id": id, "content": content, "single_word": false,
                "lstrip": false, "rstrip": false, "normalized": false, "special": true
            })
        })
        .collect();
    json!({"version": "2.0", "model": model, "added_tokens": added}).to_string()
}

fn encode(tokenizer: &PipelineTokenizer, text: &str) -> Vec<u32> {
    tokenizer
        .encode(text, &EncodeOptions::no_specials())
        .wait()
        .unwrap()[0]
        .ids()
        .iter()
        .map(|token| token.id())
        .collect()
}

fn assert_distinct_ids(tokenizer: &PipelineTokenizer, model_id: u32, new_id: u32) {
    assert_eq!(tokenizer.token_to_id("a"), Some(model_id));
    assert_eq!(tokenizer.token_to_id("<new>"), Some(new_id));
    assert_eq!(tokenizer.id_to_token(model_id).as_deref(), Some("a"));
    assert_eq!(tokenizer.decode(&[model_id], false).unwrap(), "a");
    assert_eq!(tokenizer.decode(&[new_id], false).unwrap(), "<new>");
    assert_eq!(encode(tokenizer, "a"), [model_id]);
    assert_eq!(encode(tokenizer, "<new>"), [new_id]);
    let vocab = tokenizer.get_vocab(true);
    assert_eq!(vocab.values().collect::<HashSet<_>>().len(), vocab.len());
    assert_eq!(tokenizer.get_vocab_size(false), 2);
    assert_eq!(tokenizer.get_vocab_size(true), 3);
}

fn sparse_round_trip(kind: &str) {
    for id in [2, 7] {
        let tokenizer = tk_serialize::from_json(&config(model(kind, id), &["<new>"])).unwrap();
        assert_distinct_ids(&tokenizer, id, id + 1);
        let restored = tk_serialize::from_json(&tk_serialize::to_json(&tokenizer).unwrap()).unwrap();
        assert_distinct_ids(&restored, id, id + 1);
    }
}

#[test]
fn sparse_bpe_ids_do_not_collide_with_added_tokens() {
    sparse_round_trip("BPE");
}

#[test]
fn sparse_wordpiece_ids_do_not_collide_with_added_tokens() {
    sparse_round_trip("WordPiece");
}

#[test]
fn sparse_wordlevel_ids_do_not_collide_with_added_tokens() {
    sparse_round_trip("WordLevel");
}

#[test]
fn dense_vocabularies_keep_their_added_ids() {
    for kind in ["BPE", "WordPiece", "WordLevel"] {
        let tokenizer = tk_serialize::from_json(&config(model(kind, 1), &["<new>"])).unwrap();
        assert_distinct_ids(&tokenizer, 1, 2);
    }
}

#[test]
fn registering_model_tokens_reuses_their_ids_in_either_order() {
    for kind in ["BPE", "WordPiece", "WordLevel"] {
        for tokens in [["a", "<new>"], ["<new>", "a"]] {
            let tokenizer = tk_serialize::from_json(&config(model(kind, 2), &tokens)).unwrap();
            assert_distinct_ids(&tokenizer, 2, 3);
            assert_eq!(tokenizer.decode(&[2, 3], true).unwrap(), "");
        }
    }
}

#[test]
fn unigram_dense_ids_are_unchanged() {
    let unigram = json!({
        "type": "Unigram", "unk_id": 0, "byte_fallback": false,
        "vocab": [["<unk>", 0.0], ["a", -1.0]]
    });
    let tokenizer = tk_serialize::from_json(&config(unigram, &["<new>"])).unwrap();
    assert_distinct_ids(&tokenizer, 1, 2);
}

#[test]
fn an_empty_model_starts_added_ids_at_zero() {
    let empty = json!({"type": "WordLevel", "unk_token": "<unk>", "vocab": {}});
    let tokenizer = tk_serialize::from_json(&config(empty, &["<new>"])).unwrap();
    assert_eq!(encode(&tokenizer, "<new>"), [0]);
    assert_eq!(tokenizer.decode(&[0], false).unwrap(), "<new>");
    assert_eq!(tokenizer.get_vocab_size(true), 1);
}

#[test]
fn the_last_available_id_can_be_assigned_without_wrapping() {
    let text = config(model("WordLevel", u32::MAX - 1), &["<new>"]);
    let tokenizer = tk_serialize::from_json(&text).unwrap();
    assert_distinct_ids(&tokenizer, u32::MAX - 1, u32::MAX);
    let restored = tk_serialize::from_json(&tk_serialize::to_json(&tokenizer).unwrap()).unwrap();
    assert_distinct_ids(&restored, u32::MAX - 1, u32::MAX);
}

#[test]
fn exhausted_model_ids_still_allow_existing_tokens_but_reject_new_ones() {
    for tokens in [vec![], vec!["a"]] {
        let tokenizer =
            tk_serialize::from_json(&config(model("WordLevel", u32::MAX), &tokens)).unwrap();
        assert_eq!(encode(&tokenizer, "a"), [u32::MAX]);
        assert_eq!(tokenizer.decode(&[u32::MAX], false).unwrap(), "a");
    }
    for (id, tokens) in [
        (u32::MAX, vec!["<new>"]),
        (u32::MAX - 1, vec!["<new>", "<other>"]),
    ] {
        let result = tk_serialize::from_json(&config(model("WordLevel", id), &tokens));
        let Err(error) = result else {
            panic!("an exhausted model must reject allocation of a new token ID");
        };
        assert!(error.to_string().contains("exhaust"), "{error}");
    }
}
