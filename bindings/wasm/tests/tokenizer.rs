// JsError can only be built inside a wasm runtime, so a native `cargo test` would panic on it.
#![cfg(target_arch = "wasm32")]

use tokenizers_wasm::Tokenizer;
use wasm_bindgen::JsValue;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_browser);

fn load(json: &str) -> Tokenizer {
    Tokenizer::from_json(json).map_err(JsValue::from).unwrap()
}

#[wasm_bindgen_test]
fn loads_bpe() {
    load(include_str!("../data/gpt2.json"));
    load(include_str!("../data/llama-3-tokenizer.json"));
}

#[wasm_bindgen_test]
fn loads_wordpiece() {
    load(include_str!("../data/bert-base-uncased.json"));
}

#[wasm_bindgen_test]
fn loads_unigram() {
    load(include_str!("../data/t5-base.json"));
}

#[wasm_bindgen_test]
fn rejects_config_without_model() {
    assert!(Tokenizer::from_json("{}").is_err());
}

#[wasm_bindgen_test]
fn rejects_invalid_json() {
    assert!(Tokenizer::from_json("not json").is_err());
}
