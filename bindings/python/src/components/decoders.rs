use std::sync::Arc;

use pyo3::prelude::*;
use serde_json::Value;

use crate::spec::ComponentKey;

use super::{Component, ComponentKind, ComponentSpec};

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

    fn from_spec(py: Python<'_>, spec: Arc<ComponentSpec>) -> PyResult<Py<Self>> {
        match spec.json().get("type").and_then(Value::as_str) {
            Some("ByteFallback") => Self::subclass_from_spec(py, spec, ByteFallback),
            Some("Fuse") => Self::subclass_from_spec(py, spec, Fuse),
            _ => Py::new(py, Self::make_init(spec)),
        }
    }
}

/// Turns each run of `<0xNN>` byte tokens into the text those bytes encode in UTF-8. A run
/// that is not valid UTF-8 becomes one `U+FFFD` per byte.
#[pyclass(frozen, extends = Decoder, module = "tokenizers.decoders")]
pub struct ByteFallback;

#[pymethods]
impl ByteFallback {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        Decoder::make_init(ComponentSpec::of_type("ByteFallback")).add_subclass(Self)
    }
}

/// Joins every token into one string, with no separator.
#[pyclass(frozen, extends = Decoder, module = "tokenizers.decoders")]
pub struct Fuse;

#[pymethods]
impl Fuse {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        Decoder::make_init(ComponentSpec::of_type("Fuse")).add_subclass(Self)
    }
}

#[pymodule]
pub mod decoders {
    #[pymodule_export]
    use super::{ByteFallback, Decoder, Fuse};
}
