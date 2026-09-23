//! The JavaScript surface of `Tokenizer`. Tokenization itself is tested in the core crates, so these
//! tests compare the wrapper with the core instead of pinning ids.

// JsError can only be built inside a wasm runtime, so a native `cargo test` would panic on it.
#![cfg(target_arch = "wasm32")]

use tk_encode::pipeline::{EncodeOptions as CoreOptions, Override, PipelineTokenizer};
use tk_encode::{PaddingDirection, PaddingParams, PaddingStrategy};
use tokenizers_wasm::{EncodeOptions, Tokenizer};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_browser);

const BERT: &str = include_str!("../data/bert-base-uncased.json");
const TEXT: &str = "Hello world";

fn load_tokenizer_js(json: &str) -> Tokenizer {
    Tokenizer::from_json(json).map_err(JsValue::from).unwrap()
}

fn load_tokenizer_rust(json: &str) -> PipelineTokenizer {
    tk_serialize::from_json(&tk_convert::canonicalize_str(json).unwrap()).unwrap()
}

fn encode(tokenizer: &PipelineTokenizer, options: &CoreOptions) -> Vec<u32> {
    let encodings = tokenizer.encode(TEXT, options).wait().unwrap();
    encodings[0].ids().iter().map(|token| token.id()).collect()
}

fn parse_options(json: &str) -> EncodeOptions {
    js_sys::JSON::parse(json).unwrap().unchecked_into()
}

#[wasm_bindgen_test]
fn test_from_json_error() {
    assert!(Tokenizer::from_json("not json").is_err());
    assert!(Tokenizer::from_json("{}").is_err());
}

#[wasm_bindgen_test]
fn test_encode() {
    let expected = encode(&load_tokenizer_rust(BERT), &CoreOptions::default());
    let actual = load_tokenizer_js(BERT)
        .encode(TEXT, None)
        .map_err(JsValue::from)
        .unwrap();
    assert_eq!(actual, expected);
}

#[wasm_bindgen_test]
fn test_encode_options() {
    let options = CoreOptions {
        add_special_tokens: false,
        padding: Override::With(PaddingParams {
            strategy: PaddingStrategy::Fixed(8),
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "[PAD]".to_owned(),
        }),
        ..CoreOptions::default()
    };
    let expected = encode(&load_tokenizer_rust(BERT), &options);
    let actual = load_tokenizer_js(BERT)
        .encode(
            TEXT,
            Some(parse_options(
                r#"{"addSpecialTokens":false,"padding":{"length":8}}"#,
            )),
        )
        .map_err(JsValue::from)
        .unwrap();
    assert_eq!(actual, expected);
}

const IDS: [u32; 4] = [101, 7592, 2088, 102];

// `None` checks the default: special tokens are skipped, as in the Python binding.
#[wasm_bindgen_test]
fn test_decode() {
    let rust_tokenizer = load_tokenizer_rust(BERT);
    let js_tokenizer = load_tokenizer_js(BERT);
    for (flag, skip) in [(None, true), (Some(false), false), (Some(true), true)] {
        let expected = rust_tokenizer.decode(&IDS, skip).unwrap();
        let actual = js_tokenizer
            .decode(&IDS, flag)
            .map_err(JsValue::from)
            .unwrap();
        assert_eq!(actual, expected);
    }
}

// `None` checks the default: special tokens are kept, as in the Python binding.
#[wasm_bindgen_test]
fn test_decode_tokens() {
    let rust_tokenizer = load_tokenizer_rust(BERT);
    let js_tokenizer = load_tokenizer_js(BERT);
    for (flag, skip) in [(None, false), (Some(false), false), (Some(true), true)] {
        let expected = rust_tokenizer.decode_tokens(&IDS, skip);
        let actual = js_tokenizer.decode_tokens(&IDS, flag);
        assert_eq!(actual, expected);
    }
}
