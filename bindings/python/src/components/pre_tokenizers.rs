//! Python classes definitions for pre-tokenizer components.
//! Generated from the definitions in `tk::schemas`.

use super::macros::component_module;

tk::pre_tokenizer_schemas!(component_module! {
    kind: PreTokenizer,
    module: pre_tokenizers,
    python_module: "tokenizers.pre_tokenizers",
    tokenizer_field: pre_tokenizer,
    items_key: "pretokenizers",
});
