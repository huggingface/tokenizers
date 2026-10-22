use pyo3::prelude::*;

use crate::tokenizer::ComponentKey;

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
    const JSON_KEY: ComponentKey = ComponentKey::PreTokenizer;
}

#[pymodule]
pub mod pre_tokenizers {
    #[pymodule_export]
    use super::PreTokenizer;
}
