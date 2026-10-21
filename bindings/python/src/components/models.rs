use std::sync::Arc;

use pyo3::prelude::*;
use serde_json::Value;

use crate::spec::ComponentKey;

use super::{Component, ComponentKind, ComponentSpec, JsonFields};

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

    fn from_spec(py: Python<'_>, spec: Arc<ComponentSpec>) -> PyResult<Py<Self>> {
        match spec.json().get("type").and_then(Value::as_str) {
            Some("BPE") => Self::subclass_from_spec(py, spec, Bpe),
            _ => Py::new(py, Self::make_init(spec)),
        }
    }
}

/// A byte-pair encoding model.
#[pyclass(frozen, extends = Model, name = "BPE", module = "tokenizers.models")]
pub struct Bpe;

#[pymethods]
impl Bpe {
    /// Whether a word found whole in the vocab skips the merges.
    #[getter]
    fn ignore_merges(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(slf.field::<Option<bool>>("ignore_merges")?.unwrap_or(false))
    }

    #[setter]
    fn set_ignore_merges(slf: &Bound<'_, Self>, ignore_merges: bool) {
        slf.set_field("ignore_merges", ignore_merges);
    }
}

#[pymodule]
pub mod models {
    #[pymodule_export]
    use super::{Bpe, Model};
}
