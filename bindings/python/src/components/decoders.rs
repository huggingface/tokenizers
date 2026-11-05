use pyo3::prelude::*;

use crate::components::component_spec::ComponentSpec;
use crate::tokenizer::TokenizerSpec;

use super::component_sequence::ComponentSequence;
use super::fields::{arg, value};
use super::macros::{component_classes, component_kind, sequence_class};
use super::{Component, ComponentKind, ConcreteClasses, JsonFieldsExt, check_choice};

component_kind! {
    kind: Decoder,
    python_module: "tokenizers.decoders",
    tokenizer_field: decoder,
    sequence_item_key: Some("decoders"),
}

sequence_class! {
    kind: Decoder,
    python_module: "tokenizers.decoders",
}

tk::decoder_schemas!(component_classes! {
    kind: Decoder,
    module: decoders,
    python_module: "tokenizers.decoders",
});
