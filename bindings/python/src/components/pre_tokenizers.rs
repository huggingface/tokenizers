use std::sync::Arc;

use pyo3::prelude::*;
use serde_json::Value;

use crate::spec::ComponentKey;

use super::{Component, ComponentKind, ComponentSequence, ComponentSpec};

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
            Some("Sequence") => Self::subclass_from_spec(py, spec, Sequence),
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

/// Applies each pre-tokenizer in turn, each one splitting the pieces the previous one made. Items read from it are copies: assign `seq[i] = item` to change one.
#[pyclass(frozen, extends = PreTokenizer, module = "tokenizers.pre_tokenizers")]
pub struct Sequence;

impl ComponentSequence for Sequence {
    type Kind = PreTokenizer;
    const ITEMS_KEY: &'static str = "pretokenizers";
}

#[pymethods]
impl Sequence {
    #[new]
    fn new(pre_tokenizers: Vec<Bound<'_, PreTokenizer>>) -> PyClassInitializer<Self> {
        Self::from_items(pre_tokenizers, Self)
    }

    fn __getitem__(slf: &Bound<'_, Self>, index: usize) -> PyResult<Py<PreTokenizer>> {
        Self::get(slf, index)
    }

    fn __setitem__(
        slf: &Bound<'_, Self>,
        index: usize,
        item: Bound<'_, PreTokenizer>,
    ) -> PyResult<()> {
        Self::set(slf, index, &item)
    }
}

#[pymodule]
pub mod pre_tokenizers {
    #[pymodule_export]
    use super::{
        BertPreTokenizer, PreTokenizer, Sequence, UnicodeScripts, Whitespace, WhitespaceSplit,
    };
}
