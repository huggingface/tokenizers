#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;
use std::rc::Rc;

use js_sys::{JSON, Promise, Reflect};
use tk_encode::pipeline::{EncodeOptions, PipelineTokenizer};
use tokenizers_wasm::Tokenizer;
use tsify::Ts;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};
use web_sys::{Headers, RequestInit, Response};

wasm_bindgen_test_configure!(run_in_browser);

const BERT: &str = include_str!("../data/bert-base-uncased.json");
const TEXT: &str = "Hello world";

// JsError has no Debug impl, so `unwrap` cannot be called on it directly.
fn unwrap<T>(result: Result<T, impl Into<JsValue>>) -> T {
    result.map_err(Into::into).unwrap()
}

fn core_encode() -> Vec<u32> {
    let tokenizer: PipelineTokenizer =
        tk_serialize::from_json(&tk_convert::canonicalize_str(BERT).unwrap()).unwrap();
    let encodings = tokenizer
        .encode(TEXT, &EncodeOptions::default())
        .wait()
        .unwrap();
    encodings[0].ids().iter().map(|token| token.id()).collect()
}

#[wasm_bindgen_test]
fn test_from_json() {
    let tokenizer = unwrap(Tokenizer::from_json(BERT));
    assert_eq!(unwrap(tokenizer.encode(TEXT, None)), core_encode());
}

#[wasm_bindgen_test]
fn test_from_json_error() {
    assert!(Tokenizer::from_json("not json").is_err());
    assert!(Tokenizer::from_json("{}").is_err());
}

/// Calls `from_pretrained` with `options` and a fake `fetch` that serves the BERT fixture, and
/// returns the URL and `Authorization` header of the request it made.
async fn from_pretrained(options: &str) -> (String, Option<String>) {
    let request = Rc::new(RefCell::new(None));
    let recorded = request.clone();
    let fetch = Closure::<dyn FnMut(String, RequestInit) -> Promise<Response>>::new(
        move |url, init: RequestInit| {
            let headers: Headers = init.get_headers().unchecked_into();
            *recorded.borrow_mut() = Some((url, headers.get("Authorization").unwrap()));
            Promise::resolve(&Response::new_with_opt_str(Some(BERT)).unwrap())
        },
    );
    let options = JSON::parse(options).unwrap();
    Reflect::set(&options, &"fetch".into(), fetch.as_ref()).unwrap();

    unwrap(
        Tokenizer::from_pretrained(
            "google-bert/bert-base-uncased".to_owned(),
            Some(Ts::new_unchecked(options)),
        )
        .await,
    );
    request.take().unwrap()
}

#[wasm_bindgen_test]
async fn test_from_pretrained() {
    let (url, authorization) = from_pretrained(
        r#"{"hubUrl": "https://hub.test/", "revision": "refs/pr/1", "subfolder": "tok", "token": "hf_secret"}"#,
    )
    .await;

    assert_eq!(
        url,
        "https://hub.test/google-bert/bert-base-uncased/resolve/refs%2Fpr%2F1/tok/tokenizer.json"
    );
    assert_eq!(authorization.as_deref(), Some("Bearer hf_secret"));
}
