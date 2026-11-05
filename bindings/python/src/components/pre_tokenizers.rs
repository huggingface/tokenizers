use pyo3::prelude::*;

use crate::components::component_spec::ComponentSpec;
use crate::tokenizer::TokenizerSpec;

use super::component_sequence::ComponentSequence;
use super::fields::{arg, value};
use super::macros::{component_classes, sequence_class};
use super::{Component, ComponentKind, JsonFieldsExt, check_choice};

sequence_class! {
    kind: PreTokenizer,
    python_module: "tokenizers.pre_tokenizers",
}

tk::pre_tokenizer_schemas!(component_classes! {
    kind: PreTokenizer,
    module: pre_tokenizers,
    python_module: "tokenizers.pre_tokenizers",
});

/// Base class for all pre-tokenizers
#[pyclass(frozen, subclass, extends = Component, module = "tokenizers.pre_tokenizers")]
#[derive(Default)]
pub struct PreTokenizer;

#[pymethods]
impl PreTokenizer {
    /// Rebuilds a component from its `__getstate__`
    #[staticmethod]
    fn _unpickle(py: Python<'_>, state: &[u8]) -> PyResult<Py<Self>> {
        Self::from_state(py, state)
    }
}

impl ComponentKind for PreTokenizer {
    const SEQUENCE_ITEM_KEY: Option<&'static str> = Some("pretokenizers");

    fn tokenizer_field_mut(spec: &mut TokenizerSpec) -> &mut Option<ComponentSpec> {
        &mut spec.pre_tokenizer
    }

    fn from_spec(py: Python<'_>, spec: ComponentSpec) -> PyResult<Py<Self>> {
        match &spec {
            ComponentSpec::Sequence(_) => Self::concrete_component(py, spec, Sequence),
            ComponentSpec::Single(single) => {
                let json_type = single.json_type();
                from_json_type(py, spec, json_type.as_deref())
            }
        }
    }
}
