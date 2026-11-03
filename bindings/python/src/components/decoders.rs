use pyo3::prelude::*;

use crate::components::component_spec::ComponentSpec;

use crate::tokenizer::TokenizerSpec;

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
    const ITEMS_KEY: Option<&'static str> = Some("decoders");

    const FIELD: fn(&mut TokenizerSpec) -> &mut Option<ComponentSpec> = |spec| &mut spec.decoder;

    fn from_spec(py: Python<'_>, spec: ComponentSpec) -> PyResult<Py<Self>> {
        Self::default_component(py, spec)
    }
}

#[pymodule]
pub mod decoders {
    #[pymodule_export]
    use super::Decoder;
}
