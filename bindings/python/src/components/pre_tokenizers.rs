use std::sync::Arc;

use pyo3::prelude::*;
use serde_json::Value;

use crate::spec::ComponentKey;

use super::{Component, ComponentKind, ComponentSpec};

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

    fn from_spec(py: Python<'_>, spec: Arc<ComponentSpec>) -> PyResult<Py<Self>> {
        match spec.json().get("type").and_then(Value::as_str) {
            Some("Whitespace") => Self::subclass_from_spec(py, spec, Whitespace),
            Some("WhitespaceSplit") => Self::subclass_from_spec(py, spec, WhitespaceSplit),
            Some("BertPreTokenizer") => Self::subclass_from_spec(py, spec, BertPreTokenizer),
            Some("UnicodeScripts") => Self::subclass_from_spec(py, spec, UnicodeScripts),
            _ => Py::new(py, Self::make_init(spec)),
        }
    }
}

/// Splits the text into runs of word characters and runs of punctuation, like the regex
/// `\w+|[^\w\s]+`. Whitespace is dropped.
#[pyclass(frozen, extends = PreTokenizer, module = "tokenizers.pre_tokenizers")]
pub struct Whitespace;

#[pymethods]
impl Whitespace {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        PreTokenizer::make_init(ComponentSpec::of_type("Whitespace")).add_subclass(Self)
    }
}

/// Splits the text on whitespace, like `str.split` with no arguments.
#[pyclass(frozen, extends = PreTokenizer, module = "tokenizers.pre_tokenizers")]
pub struct WhitespaceSplit;

#[pymethods]
impl WhitespaceSplit {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        PreTokenizer::make_init(ComponentSpec::of_type("WhitespaceSplit")).add_subclass(Self)
    }
}

/// Splits the text on whitespace, and isolates each punctuation character, like BERT.
#[pyclass(frozen, extends = PreTokenizer, module = "tokenizers.pre_tokenizers")]
pub struct BertPreTokenizer;

#[pymethods]
impl BertPreTokenizer {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        PreTokenizer::make_init(ComponentSpec::of_type("BertPreTokenizer")).add_subclass(Self)
    }
}

/// Splits the text where the Unicode script changes, like SentencePiece. Hiragana and
/// Katakana count as Han.
#[pyclass(frozen, extends = PreTokenizer, module = "tokenizers.pre_tokenizers")]
pub struct UnicodeScripts;

#[pymethods]
impl UnicodeScripts {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        PreTokenizer::make_init(ComponentSpec::of_type("UnicodeScripts")).add_subclass(Self)
    }
}

#[pymodule]
pub mod pre_tokenizers {
    #[pymodule_export]
    use super::{BertPreTokenizer, PreTokenizer, UnicodeScripts, Whitespace, WhitespaceSplit};
}
