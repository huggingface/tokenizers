use tk_encode::pipeline::EncodeOptions;
use tk_encode::tokenizer::PaddingStrategy;

#[test]
fn padding_declared_in_json_is_applied() {
    let config = r#"{
        "version": "2.0",
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "unk_token": "<unk>", "vocab": {"<unk>": 0, "a": 1, "b": 2, "<pad>": 3}},
        "padding": {"strategy": {"Fixed": 4}, "direction": "Right", "pad_to_multiple_of": null, "pad_id": 3, "pad_type_id": 0, "pad_token": "<pad>"},
        "truncation": null
    }"#;
    let tok = tk_serialize::from_json(config).expect("read");
    let params = tok
        .get_padding()
        .expect("padding should be loaded from json");
    assert_eq!(params.strategy, PaddingStrategy::Fixed(4));
    assert_eq!(params.pad_id, 3);

    let ids: Vec<u32> = tok
        .encode("a b", &EncodeOptions::no_specials())
        .wait()
        .unwrap()
        .remove(0)
        .ids()
        .iter()
        .map(|t| t.id())
        .collect();
    assert_eq!(
        ids,
        vec![1, 2, 3, 3],
        "config Fixed padding must apply under InheritConfig; got {ids:?}"
    );
}
