use serde_json::Value;
use tk_convert::TokenizerJson;

const LLAMA_3: &str = include_str!("../../data/llama-3-tokenizer.json");

fn parse(text: &str) -> Result<TokenizerJson, String> {
    TokenizerJson::from_value(serde_json::from_str(text).unwrap()).map_err(|e| e.to_string())
}

#[test]
fn round_trip() {
    let v2 = r#"{"version": "2.0", "truncation": null, "padding": null,
        "role_to_token": {"bos_token": "<s>"}, "added_tokens": [], "normalizer": null,
        "pre_tokenizer": null, "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "vocab": {"<s>": 0}, "unk_token": "<s>"}}"#;
    for text in [LLAMA_3, v2] {
        let written = Value::Object(parse(text).unwrap().into_object());
        assert_eq!(written, serde_json::from_str::<Value>(text).unwrap());
    }
}

#[test]
fn fills_missing_model_type() {
    let file = parse(r#"{"model": {"vocab": {"a": 0}, "merges": []}}"#).unwrap();
    assert_eq!(file.model["type"], "BPE");
}

#[test]
fn refuse_unknown_keys() {
    for (text, error) in [
        (r#"{"model": {}, "extra": 1}"#, "`extra`"),
        (r#"{"version": "2.0", "model": {}, "extra": 1}"#, "`extra`"),
        // role_to_token only exists from 2.0 on.
        (
            r#"{"version": "1.0", "role_to_token": {}, "model": {}}"#,
            "`role_to_token`",
        ),
        (r#"{"version": "3.0", "model": {}}"#, "version \"3.0\""),
        (r#"{"normalizer": [], "model": {}}"#, "`normalizer` must be"),
    ] {
        let message = parse(text).unwrap_err();
        assert!(message.contains(error), "{text}: {message}");
    }
}
