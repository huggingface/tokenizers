use super::macros::component_module;

tk::decoder_schemas!(component_module! {
    kind: Decoder,
    module: decoders,
    python_module: "tokenizers.decoders",
    tokenizer_field: decoder,
    items_key: "decoders",
});
