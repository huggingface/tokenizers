use pyo3::prelude::*;

use crate::components::component_spec::ComponentSpec;
use crate::tokenizer::TokenizerSpec;

use super::component_sequence::ComponentSequence;
use super::fields::{arg, value};
use super::macros::{component_classes, component_kind, sequence_class};
use super::{Component, ComponentKind, ConcreteClasses, JsonFieldsExt, check_choice};

component_kind! {
    kind: Normalizer,
    python_module: "tokenizers.normalizers",
    tokenizer_field: normalizer,
    sequence_item_key: Some("normalizers"),
}

sequence_class! {
    kind: Normalizer,
    python_module: "tokenizers.normalizers",
}

tk::normalizer_schemas!(component_classes! {
    kind: Normalizer,
    module: normalizers,
    python_module: "tokenizers.normalizers",
});
