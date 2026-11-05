use pyo3::prelude::*;

use crate::components::component_spec::ComponentSpec;
use crate::tokenizer::TokenizerSpec;

use super::component_sequence::ComponentSequence;
use super::macros::{component_classes, component_kind, sequence_class};
use super::{Component, ComponentKind, ConcreteClasses};

component_kind! {
    kind: PreTokenizer,
    python_module: "tokenizers.pre_tokenizers",
    tokenizer_field: pre_tokenizer,
    sequence_item_key: Some("pretokenizers"),
}

sequence_class! {
    kind: PreTokenizer,
    python_module: "tokenizers.pre_tokenizers",
}

tk::pre_tokenizer_schemas!(component_classes! {
    kind: PreTokenizer,
    module: pre_tokenizers,
    python_module: "tokenizers.pre_tokenizers",
});
