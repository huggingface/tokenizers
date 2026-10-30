use std::sync::Arc;

use pyo3::prelude::*;
use serde_json::{Value, json};

use crate::error::err;
use crate::regex::Pattern;
use crate::tokenizer::{ComponentKey, ComponentSpec};

use super::{Component, ComponentKind, ComponentSequence, ComponentType};

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
            Some("Whitespace") => Whitespace::from_spec(py, spec),
            Some("WhitespaceSplit") => WhitespaceSplit::from_spec(py, spec),
            Some("BertPreTokenizer") => BertPreTokenizer::from_spec(py, spec),
            Some("UnicodeScripts") => UnicodeScripts::from_spec(py, spec),
            Some("Split") => Split::from_spec(py, spec),
            Some("Digits") => Digits::from_spec(py, spec),
            Some("Punctuation") => Punctuation::from_spec(py, spec),
            Some("Sequence") => Sequence::from_spec(py, spec),
            _ => Self::default_component(py, spec),
        }
    }
}

/// Splits the text into runs of word characters and runs of punctuation, like the regex
/// `\w+|[^\w\s]+`. Whitespace is dropped.
#[pyclass(frozen, extends = PreTokenizer, module = "tokenizers.pre_tokenizers")]
#[derive(Default)]
pub struct Whitespace;

impl ComponentType for Whitespace {}

#[pymethods]
impl Whitespace {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        let Value::Object(json) = json!({"type": "Whitespace"}) else {
            unreachable!("json! builds an object from an object literal")
        };
        Self::initializer(ComponentSpec::new(json))
    }
}

/// Splits the text on whitespace, like `str.split` with no arguments.
#[pyclass(frozen, extends = PreTokenizer, module = "tokenizers.pre_tokenizers")]
#[derive(Default)]
pub struct WhitespaceSplit;

impl ComponentType for WhitespaceSplit {}

#[pymethods]
impl WhitespaceSplit {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        let Value::Object(json) = json!({"type": "WhitespaceSplit"}) else {
            unreachable!("json! builds an object from an object literal")
        };
        Self::initializer(ComponentSpec::new(json))
    }
}

/// Splits the text on whitespace, and isolates each punctuation character, like BERT.
#[pyclass(frozen, extends = PreTokenizer, module = "tokenizers.pre_tokenizers")]
#[derive(Default)]
pub struct BertPreTokenizer;

impl ComponentType for BertPreTokenizer {}

#[pymethods]
impl BertPreTokenizer {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        let Value::Object(json) = json!({"type": "BertPreTokenizer"}) else {
            unreachable!("json! builds an object from an object literal")
        };
        Self::initializer(ComponentSpec::new(json))
    }
}

/// Splits the text where the Unicode script changes, like SentencePiece. Hiragana and
/// Katakana count as Han.
#[pyclass(frozen, extends = PreTokenizer, module = "tokenizers.pre_tokenizers")]
#[derive(Default)]
pub struct UnicodeScripts;

impl ComponentType for UnicodeScripts {}

#[pymethods]
impl UnicodeScripts {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        let Value::Object(json) = json!({"type": "UnicodeScripts"}) else {
            unreachable!("json! builds an object from an object literal")
        };
        Self::initializer(ComponentSpec::new(json))
    }
}

/// The `tokenizer.json` spelling of a `behavior` argument: what happens to the delimiters.
fn delimiter_behavior(behavior: &str) -> PyResult<&'static str> {
    match behavior {
        "removed" => Ok("Removed"),
        "isolated" => Ok("Isolated"),
        "merged_with_previous" => Ok("MergedWithPrevious"),
        "merged_with_next" => Ok("MergedWithNext"),
        "contiguous" => Ok("Contiguous"),
        _ => Err(err(format!(
            "behavior must be one of \"removed\", \"isolated\", \"merged_with_previous\", \
             \"merged_with_next\" or \"contiguous\", not {behavior:?}"
        ))),
    }
}

/// Splits the text on every match of `pattern`, the matches being the delimiters. A `str` pattern
/// is a literal. `invert` splits on what does not match instead.
#[pyclass(frozen, extends = PreTokenizer, module = "tokenizers.pre_tokenizers")]
#[derive(Default)]
pub struct Split;

impl ComponentType for Split {}

#[pymethods]
impl Split {
    /// `behavior` is one of `"removed"`, `"isolated"`, `"merged_with_previous"`,
    /// `"merged_with_next"` or `"contiguous"`.
    #[new]
    #[pyo3(signature = (pattern, behavior, invert=false))]
    fn new(pattern: Pattern, behavior: &str, invert: bool) -> PyResult<PyClassInitializer<Self>> {
        let Value::Object(json) = json!({
            "type": "Split",
            "pattern": Value::from(pattern),
            "behavior": delimiter_behavior(behavior)?,
            "invert": invert,
        }) else {
            unreachable!("json! builds an object from an object literal")
        };
        Ok(Self::initializer(ComponentSpec::new(json)))
    }
}

/// Splits the text around runs of digits, or around each digit with `individual_digits`.
#[pyclass(frozen, extends = PreTokenizer, module = "tokenizers.pre_tokenizers")]
#[derive(Default)]
pub struct Digits;

impl ComponentType for Digits {}

#[pymethods]
impl Digits {
    #[new]
    #[pyo3(signature = (individual_digits=false))]
    fn new(individual_digits: bool) -> PyClassInitializer<Self> {
        let Value::Object(json) = json!({"type": "Digits", "individual_digits": individual_digits})
        else {
            unreachable!("json! builds an object from an object literal")
        };
        Self::initializer(ComponentSpec::new(json))
    }
}

/// Splits the text on each punctuation character. `behavior` is that of `Split`.
#[pyclass(frozen, extends = PreTokenizer, module = "tokenizers.pre_tokenizers")]
#[derive(Default)]
pub struct Punctuation;

impl ComponentType for Punctuation {}

#[pymethods]
impl Punctuation {
    #[new]
    #[pyo3(signature = (behavior="isolated"))]
    fn new(behavior: &str) -> PyResult<PyClassInitializer<Self>> {
        let Value::Object(json) =
            json!({"type": "Punctuation", "behavior": delimiter_behavior(behavior)?})
        else {
            unreachable!("json! builds an object from an object literal")
        };
        Ok(Self::initializer(ComponentSpec::new(json)))
    }
}

/// Applies each pre-tokenizer in turn, each one splitting the pieces the previous one made. Items read from it are copies: assign `seq[i] = item` to change one.
#[pyclass(frozen, extends = PreTokenizer, module = "tokenizers.pre_tokenizers")]
#[derive(Default)]
pub struct Sequence;

impl ComponentType for Sequence {}

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
        BertPreTokenizer, Digits, PreTokenizer, Punctuation, Sequence, Split, UnicodeScripts,
        Whitespace, WhitespaceSplit,
    };
}
