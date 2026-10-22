use pyo3::prelude::*;

use crate::tokenizer::ComponentKey;

use super::{Component, ComponentKind};

/// Base class for all normalizers
#[pyclass(frozen, subclass, extends = Component, module = "tokenizers.normalizers")]
#[derive(Default)]
pub struct Normalizer;

#[pymethods]
impl Normalizer {
    /// Rebuilds a component from its `__getstate__`
    #[staticmethod]
    fn _unpickle(py: Python<'_>, state: &[u8]) -> PyResult<Py<Self>> {
        Self::from_state(py, state)
    }
}

impl ComponentKind for Normalizer {
    const JSON_KEY: ComponentKey = ComponentKey::Normalizer;
}

#[pymodule]
pub mod normalizers {
    #[pymodule_export]
    use super::Normalizer;
}
