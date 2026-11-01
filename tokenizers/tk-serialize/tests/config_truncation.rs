use tk_encode::pipeline::EncodeOptions;

#[test]
fn truncation_declared_in_json_is_applied() {
    let config = r#"{
        "version": "2.0",
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "unk_token": "<unk>", "vocab": {"<unk>": 0, "a": 1, "b": 2}},
        "padding": null,
        "truncation": {"max_length": 3, "strategy": "LongestFirst", "stride": 0, "direction": "Right"}
    }"#;
    let tok = tk_serialize::from_json(config).expect("read");
    assert!(
        tok.get_truncation().is_some(),
        "truncation should be loaded from json"
    );
    let ids: Vec<u32> = tok
        .encode("a a a a a", &EncodeOptions::no_specials())
        .wait()
        .unwrap()
        .remove(0)
        .ids()
        .iter()
        .map(|t| t.id())
        .collect();
    assert_eq!(
        ids,
        vec![1, 1, 1],
        "config truncation must apply under InheritConfig; got {ids:?}"
    );
}
