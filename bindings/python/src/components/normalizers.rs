use std::sync::Arc;

use pyo3::prelude::*;
use serde_json::Value;

use crate::spec::ComponentKey;

use super::{Component, ComponentKind, ComponentSpec};

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
            Some("Lowercase") => Self::subclass_from_spec(py, spec, Lowercase),
            Some("ByteLevel") => Self::subclass_from_spec(py, spec, ByteLevel),
            Some("StripAccents") => Self::subclass_from_spec(py, spec, StripAccents),
            Some("NFC") => Self::subclass_from_spec(py, spec, Nfc),
            Some("NFD") => Self::subclass_from_spec(py, spec, Nfd),
            Some("NFKC") => Self::subclass_from_spec(py, spec, Nfkc),
            Some("NFKD") => Self::subclass_from_spec(py, spec, Nfkd),
            Some("Nmt") => Self::subclass_from_spec(py, spec, Nmt),
            _ => Py::new(py, Self::make_init(spec)),
        }
    }
}

/// Lowercases the text, like `str.lower`.
#[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
pub struct Lowercase;

#[pymethods]
impl Lowercase {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        Normalizer::make_init(ComponentSpec::of_type("Lowercase")).add_subclass(Self)
    }
}

/// Replaces each byte of the text with its character in the GPT-2 byte-to-unicode mapping.
#[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
pub struct ByteLevel;

#[pymethods]
impl ByteLevel {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        Normalizer::make_init(ComponentSpec::of_type("ByteLevel")).add_subclass(Self)
    }
}

/// Removes combining marks. Use it after `NFD` or `NFKD` to strip accents.
#[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
pub struct StripAccents;

#[pymethods]
impl StripAccents {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        Normalizer::make_init(ComponentSpec::of_type("StripAccents")).add_subclass(Self)
    }
}

/// Unicode NFC normalization.
#[pyclass(frozen, extends = Normalizer, name = "NFC", module = "tokenizers.normalizers")]
pub struct Nfc;

#[pymethods]
impl Nfc {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        Normalizer::make_init(ComponentSpec::of_type("NFC")).add_subclass(Self)
    }
}

/// Unicode NFD normalization.
#[pyclass(frozen, extends = Normalizer, name = "NFD", module = "tokenizers.normalizers")]
pub struct Nfd;

#[pymethods]
impl Nfd {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        Normalizer::make_init(ComponentSpec::of_type("NFD")).add_subclass(Self)
    }
}

/// Unicode NFKC normalization.
#[pyclass(frozen, extends = Normalizer, name = "NFKC", module = "tokenizers.normalizers")]
pub struct Nfkc;

#[pymethods]
impl Nfkc {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        Normalizer::make_init(ComponentSpec::of_type("NFKC")).add_subclass(Self)
    }
}

/// Unicode NFKD normalization.
#[pyclass(frozen, extends = Normalizer, name = "NFKD", module = "tokenizers.normalizers")]
pub struct Nfkd;

#[pymethods]
impl Nfkd {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        Normalizer::make_init(ComponentSpec::of_type("NFKD")).add_subclass(Self)
    }
}

/// Removes control characters and replaces line breaks, tabs and some invisible characters with a
/// space, like the Google NMT pipeline.
#[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
pub struct Nmt;

#[pymethods]
impl Nmt {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        Normalizer::make_init(ComponentSpec::of_type("Nmt")).add_subclass(Self)
    }
}

#[pymodule]
pub mod normalizers {
    #[pymodule_export]
    use super::{ByteLevel, Lowercase, Nfc, Nfd, Nfkc, Nfkd, Nmt, Normalizer, StripAccents};
}
