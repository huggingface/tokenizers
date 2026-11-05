use super::macros::component_module;

tk::normalizer_schemas!(component_module! {
    kind: Normalizer,
    module: normalizers,
    python_module: "tokenizers.normalizers",
    tokenizer_field: normalizer,
    items_key: "normalizers",
});
