//! Python classes definitions for decoder components.
//! Generated from the definitions in `tk::schemas`.

use super::macros::component_module;

tk::decoder_schemas!(component_module! {
    kind: Decoder,
    module: decoders,
    python_module: "tokenizers.decoders",
    tokenizer_field: decoder,
    items_key: "decoders",
    also_exports: [crate::decode_stream::DecodeStream],
});
