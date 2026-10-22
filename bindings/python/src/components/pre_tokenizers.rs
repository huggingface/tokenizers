use std::sync::Arc;

use pyo3::prelude::*;
use serde_json::{Value, json};

use crate::error::err;
use crate::regex::Pattern;
use crate::spec::ComponentKey;

use super::{Component, ComponentKind, ComponentSequence, ComponentSpec, JsonFields};

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
            Some("ByteLevel") => Self::subclass_from_spec(py, spec, ByteLevel),
            Some("Metaspace") => Self::subclass_from_spec(py, spec, Metaspace),
            Some("Split") => Self::subclass_from_spec(py, spec, Split),
            Some("Digits") => Self::subclass_from_spec(py, spec, Digits),
            Some("Punctuation") => Self::subclass_from_spec(py, spec, Punctuation),
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
        PreTokenizer::init(json!({"type": "Whitespace"}), Self)
    }
}

/// Splits the text on whitespace, like `str.split` with no arguments.
#[pyclass(frozen, extends = PreTokenizer, module = "tokenizers.pre_tokenizers")]
pub struct WhitespaceSplit;

#[pymethods]
impl WhitespaceSplit {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        PreTokenizer::init(json!({"type": "WhitespaceSplit"}), Self)
    }
}

/// Splits the text on whitespace, and isolates each punctuation character, like BERT.
#[pyclass(frozen, extends = PreTokenizer, module = "tokenizers.pre_tokenizers")]
pub struct BertPreTokenizer;

#[pymethods]
impl BertPreTokenizer {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        PreTokenizer::init(json!({"type": "BertPreTokenizer"}), Self)
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
        PreTokenizer::init(json!({"type": "UnicodeScripts"}), Self)
    }
}

/// GPT-2's byte-level pre-tokenization: splits the text with GPT-2's regex, then replaces each
/// byte with its character in the GPT-2 byte-to-unicode mapping.
#[pyclass(frozen, extends = PreTokenizer, module = "tokenizers.pre_tokenizers")]
pub struct ByteLevel;

#[pymethods]
impl ByteLevel {
    /// `add_prefix_space` adds a space in front of the text, so the first word encodes like the
    /// others. `use_regex=False` skips the split.
    #[new]
    #[pyo3(signature = (add_prefix_space=true, trim_offsets=true, use_regex=true))]
    fn new(
        add_prefix_space: bool,
        trim_offsets: bool,
        use_regex: bool,
    ) -> PyClassInitializer<Self> {
        let json = json!({
            "type": "ByteLevel",
            "add_prefix_space": add_prefix_space,
            "trim_offsets": trim_offsets,
            "use_regex": use_regex,
        });
        PreTokenizer::init(json, Self)
    }

    /// The 256 characters of the byte-to-unicode mapping, one per byte.
    #[staticmethod]
    fn alphabet() -> Vec<char> {
        tk::pre_tokenizers::byte_level::ByteLevel::alphabet().to_vec()
    }

    #[getter]
    fn add_prefix_space(slf: &Bound<'_, Self>) -> PyResult<bool> {
        slf.field("add_prefix_space")
    }
}

/// SentencePiece's whitespace handling: replaces spaces with `replacement` and splits the text in
/// front of each one when `split` is set.
#[pyclass(frozen, extends = PreTokenizer, module = "tokenizers.pre_tokenizers")]
pub struct Metaspace;

#[pymethods]
impl Metaspace {
    /// `prepend_scheme` says when `replacement` goes in front of the text: `"always"`, `"first"`
    /// (only at the start of the input) or `"never"`.
    #[new]
    #[pyo3(signature = (replacement='▁', prepend_scheme="always", split=true))]
    fn new(
        replacement: char,
        prepend_scheme: &str,
        split: bool,
    ) -> PyResult<PyClassInitializer<Self>> {
        if !["always", "first", "never"].contains(&prepend_scheme) {
            return Err(err(format!(
                "prepend_scheme must be \"always\", \"first\" or \"never\", not {prepend_scheme:?}"
            )));
        }
        let json = json!({
            "type": "Metaspace",
            "replacement": replacement,
            "prepend_scheme": prepend_scheme,
            "split": split,
        });
        Ok(PreTokenizer::init(json, Self))
    }

    #[getter]
    fn replacement(slf: &Bound<'_, Self>) -> PyResult<char> {
        slf.field("replacement")
    }

    #[getter]
    fn split(slf: &Bound<'_, Self>) -> PyResult<bool> {
        slf.field("split")
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
pub struct Split;

#[pymethods]
impl Split {
    /// `behavior` is one of `"removed"`, `"isolated"`, `"merged_with_previous"`,
    /// `"merged_with_next"` or `"contiguous"`.
    #[new]
    #[pyo3(signature = (pattern, behavior, invert=false))]
    fn new(pattern: Pattern, behavior: &str, invert: bool) -> PyResult<PyClassInitializer<Self>> {
        let json = json!({
            "type": "Split",
            "pattern": Value::from(pattern),
            "behavior": delimiter_behavior(behavior)?,
            "invert": invert,
        });
        Ok(PreTokenizer::init(json, Self))
    }
}

/// Splits the text around runs of digits, or around each digit with `individual_digits`.
#[pyclass(frozen, extends = PreTokenizer, module = "tokenizers.pre_tokenizers")]
pub struct Digits;

#[pymethods]
impl Digits {
    #[new]
    #[pyo3(signature = (individual_digits=false))]
    fn new(individual_digits: bool) -> PyClassInitializer<Self> {
        let json = json!({"type": "Digits", "individual_digits": individual_digits});
        PreTokenizer::init(json, Self)
    }
}

/// Splits the text on each punctuation character. `behavior` is that of `Split`.
#[pyclass(frozen, extends = PreTokenizer, module = "tokenizers.pre_tokenizers")]
pub struct Punctuation;

#[pymethods]
impl Punctuation {
    #[new]
    #[pyo3(signature = (behavior="isolated"))]
    fn new(behavior: &str) -> PyResult<PyClassInitializer<Self>> {
        let json = json!({"type": "Punctuation", "behavior": delimiter_behavior(behavior)?});
        Ok(PreTokenizer::init(json, Self))
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
        BertPreTokenizer, ByteLevel, Digits, Metaspace, PreTokenizer, Punctuation, Sequence, Split,
        UnicodeScripts, Whitespace, WhitespaceSplit,
    };
}
