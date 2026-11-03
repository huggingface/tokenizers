use pyo3::prelude::*;

use crate::components::component_spec::ComponentSpec;

use crate::tokenizer::TokenizerSpec;

use super::{Component, ComponentKind};

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
    const ITEMS_KEY: Option<&'static str> = Some("pretokenizers");

    const FIELD: fn(&mut TokenizerSpec) -> &mut Option<ComponentSpec> =
        |spec| &mut spec.pre_tokenizer;

    fn from_spec(py: Python<'_>, spec: ComponentSpec) -> PyResult<Py<Self>> {
        Self::default_component(py, spec)
    }
}

#[pymodule]
pub mod pre_tokenizers {
    #[pymodule_export]
    use super::PreTokenizer;
}
