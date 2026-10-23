use std::sync::Arc;

use pyo3::prelude::*;
use serde_json::{Value, json};

use crate::tokenizer::{ComponentKey, ComponentSpec};

use super::{Component, ComponentKind, ComponentType, JsonFields};

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

    fn from_spec(py: Python<'_>, spec: Arc<ComponentSpec>) -> PyResult<Py<Self>> {
        match spec.json().get("type").and_then(Value::as_str) {
            Some("Prepend") => Prepend::from_spec(py, spec),
            _ => Normalizer::default_component(py, spec),
        }
    }
}

/// Adds `prepend` at the start of the text.
#[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
#[derive(Default)]
pub struct Prepend;

impl ComponentType for Prepend {}

#[pymethods]
impl Prepend {
    #[new]
    fn new(prepend: String) -> PyClassInitializer<Self> {
        let Value::Object(json) = json!({"type": "Prepend", "prepend": prepend}) else {
            unreachable!("json! builds an object from an object literal")
        };
        Self::initializer(ComponentSpec::new(json))
    }

    #[getter]
    fn prepend(slf: &Bound<'_, Self>) -> PyResult<String> {
        slf.field("prepend")
    }

    #[setter]
    fn set_prepend(slf: &Bound<'_, Self>, prepend: String) {
        slf.set_field("prepend", prepend);
    }
}

#[pymodule]
pub mod normalizers {
    #[pymodule_export]
    use super::{Normalizer, Prepend};
}
