use pyo3::prelude::*;

use crate::spec::ComponentKey;

use super::{Component, ComponentKind};

/// Base class for all decoders
#[pyclass(frozen, subclass, extends = Component, module = "tokenizers.decoders")]
#[derive(Default)]
pub struct Decoder;

#[pymethods]
impl Decoder {
    /// Rebuilds a component from its `__getstate__`
    #[staticmethod]
    fn _unpickle(py: Python<'_>, state: &[u8]) -> PyResult<Py<Self>> {
        Self::from_state(py, state)
    }
}

impl ComponentKind for Decoder {
    const JSON_KEY: ComponentKey = ComponentKey::Decoder;
}

#[pymodule]
pub mod decoders {
    #[pymodule_export]
    use super::Decoder;
}
