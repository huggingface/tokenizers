//! `Tokenizer.from_pretrained` against a fake `fetch` that serves the BERT fixture, so no test
//! touches the network.

#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;
use std::rc::Rc;

use js_sys::{JSON, Promise, Reflect};
use tsify::Ts;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};
use web_sys::{Headers, RequestInit, Response};

use tokenizers_web::Tokenizer;

wasm_bindgen_test_configure!(run_in_browser);

const BERT: &str = include_str!("../data/bert-base-uncased.json");

struct Request {
    url: String,
    authorization: Option<String>,
}

/// Calls `from_pretrained` with `options` and a fake `fetch`, and returns the tokenizer and the
/// request it made.
async fn from_pretrained(options: &str) -> (Tokenizer, Request) {
    let requests = Rc::new(RefCell::new(Vec::new()));
    let recorded = requests.clone();
    let fetch = Closure::<dyn FnMut(String, RequestInit) -> Promise<Response>>::new(
        move |url, init: RequestInit| {
            let headers = init.get_headers();
            let authorization = (!headers.is_undefined())
                .then(|| {
                    headers
                        .unchecked_into::<Headers>()
                        .get("Authorization")
                        .unwrap()
                })
                .flatten();
            recorded.borrow_mut().push(Request { url, authorization });
            Promise::resolve(&Response::new_with_opt_str(Some(BERT)).unwrap())
        },
    );
    let options = JSON::parse(options).unwrap();
    Reflect::set(&options, &"fetch".into(), fetch.as_ref()).unwrap();

    let tokenizer = Tokenizer::from_pretrained(
        "google-bert/bert-base-uncased".to_owned(),
        Some(Ts::new_unchecked(options)),
    )
    .await
    .map_err(|err| format!("{err:?}"))
    .unwrap();
    let request = requests.borrow_mut().pop().unwrap();
    (tokenizer, request)
}

#[wasm_bindgen_test]
async fn test_from_pretrained_defaults() {
    let (tokenizer, request) = from_pretrained("{}").await;

    let expected = Tokenizer::from_json(BERT).map_err(JsValue::from).unwrap();
    let ids = |tokenizer: &Tokenizer| {
        tokenizer
            .encode("Hello world", None)
            .map_err(JsValue::from)
            .unwrap()
    };
    assert_eq!(ids(&tokenizer), ids(&expected));
    assert_eq!(
        request.url,
        "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/tokenizer.json"
    );
    assert_eq!(request.authorization, None);
}

#[wasm_bindgen_test]
async fn test_from_pretrained_options() {
    let (_, request) = from_pretrained(
        r#"{"hubUrl": "https://hub.test/", "revision": "refs/pr/1", "subfolder": "tok", "token": "hf_secret"}"#,
    )
    .await;

    assert_eq!(
        request.url,
        "https://hub.test/google-bert/bert-base-uncased/resolve/refs%2Fpr%2F1/tok/tokenizer.json"
    );
    assert_eq!(request.authorization.as_deref(), Some("Bearer hf_secret"));
}
