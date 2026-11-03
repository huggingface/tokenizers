use pyo3::prelude::*;

use crate::components::component_spec::ComponentSpec;

use crate::tokenizer::TokenizerSpec;

use super::{Component, ComponentKind};

/// Base class for all tokenizer models
#[pyclass(frozen, subclass, extends = Component, module = "tokenizers.models")]
#[derive(Default)]
pub struct Model;

#[pymethods]
impl Model {
    /// Rebuilds a component from its `__getstate__`
    #[staticmethod]
    fn _unpickle(py: Python<'_>, state: &[u8]) -> PyResult<Py<Self>> {
        Self::from_state(py, state)
    }
}

impl ComponentKind for Model {
    const ITEMS_KEY: Option<&'static str> = None;

    const FIELD: fn(&mut TokenizerSpec) -> &mut Option<ComponentSpec> = |spec| &mut spec.model;

    fn from_spec(py: Python<'_>, spec: ComponentSpec) -> PyResult<Py<Self>> {
        Self::default_component(py, spec)
    }
}

#[pymodule]
pub mod models {
    #[pymodule_export]
    use super::Model;
}
