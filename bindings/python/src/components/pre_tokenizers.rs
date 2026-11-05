use pyo3::prelude::*;

use crate::components::component_spec::ComponentSpec;
use crate::tokenizer::TokenizerSpec;

use super::fields::py_classes;
use super::{Component, ComponentKind};

tk::pre_tokenizer_schemas!(
    py_classes! { PreTokenizer, pre_tokenizers, "tokenizers.pre_tokenizers", "Applies each pre-tokenizer in turn, each one splitting the pieces the previous one made." }
);

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
