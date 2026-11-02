use std::sync::Arc;

use pyo3::prelude::*;

use crate::tokenizer::{ComponentKey, ComponentSpec};

use super::{Component, ComponentKind};

/// Base class for all post-processors
#[pyclass(frozen, subclass, extends = Component, module = "tokenizers.processors")]
#[derive(Default)]
pub struct PostProcessor;

#[pymethods]
impl PostProcessor {
    /// Rebuilds a component from its `__getstate__`
    #[staticmethod]
    fn _unpickle(py: Python<'_>, state: &[u8]) -> PyResult<Py<Self>> {
        Self::from_state(py, state)
    }
}

impl ComponentKind for PostProcessor {
    const JSON_KEY: ComponentKey = ComponentKey::PostProcessor;

    fn from_spec(py: Python<'_>, spec: Arc<ComponentSpec>) -> PyResult<Py<Self>> {
        Self::default_component(py, spec)
    }
}

#[pymodule]
pub mod processors {
    #[pymodule_export]
    use super::PostProcessor;
}
