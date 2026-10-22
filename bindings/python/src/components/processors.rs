use std::sync::Arc;

use pyo3::prelude::*;
use serde_json::Value;

use crate::spec::ComponentKey;

use super::{Component, ComponentKind, ComponentSequence, ComponentSpec};

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
        match spec.json().get("type").and_then(Value::as_str) {
            Some("Sequence") => Self::subclass_from_spec(py, spec, Sequence),
            _ => Py::new(py, Self::make_init(spec)),
        }
    }
}

/// Applies each post-processor in turn.
#[pyclass(frozen, extends = PostProcessor, module = "tokenizers.processors")]
pub struct Sequence;

impl ComponentSequence for Sequence {
    type Kind = PostProcessor;
    const ITEMS_KEY: &'static str = "processors";
}

#[pymethods]
impl Sequence {
    #[new]
    fn new(processors: Vec<Bound<'_, PostProcessor>>) -> PyClassInitializer<Self> {
        Self::from_items(processors, Self)
    }
}

#[pymodule]
pub mod processors {
    #[pymodule_export]
    use super::{PostProcessor, Sequence};
}
