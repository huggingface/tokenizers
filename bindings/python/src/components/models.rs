use pyo3::prelude::*;

use crate::spec::ComponentKey;

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
    const JSON_KEY: ComponentKey = ComponentKey::Model;
}

#[pymodule]
pub mod models {
    #[pymodule_export]
    use super::Model;
}
