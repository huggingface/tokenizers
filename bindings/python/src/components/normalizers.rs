use std::sync::Arc;

use base64::Engine as _;
use pyo3::prelude::*;
use serde_json::{Value, json};

use crate::regex::Pattern;
use crate::spec::ComponentKey;

use super::{Component, ComponentKind, ComponentSequence, ComponentSpec, JsonFields};

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
            Some("Replace") => Self::subclass_from_spec(py, spec, Replace),
            Some("Precompiled") => Self::subclass_from_spec(py, spec, Precompiled),
            Some("BertNormalizer") => Self::subclass_from_spec(py, spec, BertNormalizer),
            Some("Strip") => Self::subclass_from_spec(py, spec, Strip),
            Some("Prepend") => Self::subclass_from_spec(py, spec, Prepend),
            Some("Sequence") => Self::subclass_from_spec(py, spec, Sequence),
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
        Normalizer::init(json!({"type": "Lowercase"}), Self)
    }
}

/// Replaces each byte of the text with its character in the GPT-2 byte-to-unicode mapping.
#[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
pub struct ByteLevel;

#[pymethods]
impl ByteLevel {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        Normalizer::init(json!({"type": "ByteLevel"}), Self)
    }
}

/// Removes combining marks. Use it after `NFD` or `NFKD` to strip accents.
#[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
pub struct StripAccents;

#[pymethods]
impl StripAccents {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        Normalizer::init(json!({"type": "StripAccents"}), Self)
    }
}

/// Unicode NFC normalization.
#[pyclass(frozen, extends = Normalizer, name = "NFC", module = "tokenizers.normalizers")]
pub struct Nfc;

#[pymethods]
impl Nfc {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        Normalizer::init(json!({"type": "NFC"}), Self)
    }
}

/// Unicode NFD normalization.
#[pyclass(frozen, extends = Normalizer, name = "NFD", module = "tokenizers.normalizers")]
pub struct Nfd;

#[pymethods]
impl Nfd {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        Normalizer::init(json!({"type": "NFD"}), Self)
    }
}

/// Unicode NFKC normalization.
#[pyclass(frozen, extends = Normalizer, name = "NFKC", module = "tokenizers.normalizers")]
pub struct Nfkc;

#[pymethods]
impl Nfkc {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        Normalizer::init(json!({"type": "NFKC"}), Self)
    }
}

/// Unicode NFKD normalization.
#[pyclass(frozen, extends = Normalizer, name = "NFKD", module = "tokenizers.normalizers")]
pub struct Nfkd;

#[pymethods]
impl Nfkd {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        Normalizer::init(json!({"type": "NFKD"}), Self)
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
        Normalizer::init(json!({"type": "Nmt"}), Self)
    }
}

/// Replaces every match of `pattern` with `content`. A `str` pattern is a literal.
#[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
pub struct Replace;

#[pymethods]
impl Replace {
    #[new]
    fn new(pattern: Pattern, content: String) -> PyClassInitializer<Self> {
        let json = json!({"type": "Replace", "pattern": Value::from(pattern), "content": content});
        Normalizer::init(json, Self)
    }
}

/// SentencePiece's normalization rules, compiled into a charsmap (T5, ALBERT, XLM-R...).
#[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
pub struct Precompiled;

#[pymethods]
impl Precompiled {
    #[new]
    fn new(precompiled_charsmap: &[u8]) -> PyClassInitializer<Self> {
        let charsmap = base64::engine::general_purpose::STANDARD.encode(precompiled_charsmap);
        let json = json!({"type": "Precompiled", "precompiled_charsmap": charsmap});
        Normalizer::init(json, Self)
    }
}

/// BERT's normalization: control characters removed, spaces around CJK characters, accents
/// stripped and the text lowercased, each optional.
#[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
pub struct BertNormalizer;

#[pymethods]
impl BertNormalizer {
    /// `strip_accents=None` strips accents when `lowercase` is set.
    #[new]
    #[pyo3(signature = (clean_text=true, handle_chinese_chars=true, strip_accents=None, lowercase=true))]
    fn new(
        clean_text: bool,
        handle_chinese_chars: bool,
        strip_accents: Option<bool>,
        lowercase: bool,
    ) -> PyClassInitializer<Self> {
        let json = json!({
            "type": "BertNormalizer",
            "clean_text": clean_text,
            "handle_chinese_chars": handle_chinese_chars,
            "strip_accents": strip_accents,
            "lowercase": lowercase,
        });
        Normalizer::init(json, Self)
    }

    #[getter]
    fn strip_accents(slf: &Bound<'_, Self>) -> PyResult<Option<bool>> {
        slf.field("strip_accents")
    }
}

/// Removes whitespace from the start and the end of the text.
#[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
pub struct Strip;

#[pymethods]
impl Strip {
    #[new]
    #[pyo3(signature = (left=true, right=true))]
    fn new(left: bool, right: bool) -> PyClassInitializer<Self> {
        let json = json!({"type": "Strip", "strip_left": left, "strip_right": right});
        Normalizer::init(json, Self)
    }
}

/// Adds `prepend` at the start of the text, unless the text is empty.
#[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
pub struct Prepend;

#[pymethods]
impl Prepend {
    #[new]
    fn new(prepend: String) -> PyClassInitializer<Self> {
        Normalizer::init(json!({"type": "Prepend", "prepend": prepend}), Self)
    }
}

/// Applies each normalizer in turn. Items read from it are copies: assign `seq[i] = item` to change one.
#[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
pub struct Sequence;

impl ComponentSequence for Sequence {
    type Kind = Normalizer;
    const ITEMS_KEY: &'static str = "normalizers";
}

#[pymethods]
impl Sequence {
    #[new]
    fn new(normalizers: Vec<Bound<'_, Normalizer>>) -> PyClassInitializer<Self> {
        Self::from_items(normalizers, Self)
    }

    fn __len__(slf: &Bound<'_, Self>) -> usize {
        Self::len(slf)
    }

    fn __getitem__(slf: &Bound<'_, Self>, index: usize) -> PyResult<Py<Normalizer>> {
        Self::get(slf, index)
    }

    fn __setitem__(
        slf: &Bound<'_, Self>,
        index: usize,
        item: Bound<'_, Normalizer>,
    ) -> PyResult<()> {
        Self::set(slf, index, &item)
    }
}

#[pymodule]
pub mod normalizers {
    #[pymodule_export]
    use super::{
        BertNormalizer, ByteLevel, Lowercase, Nfc, Nfd, Nfkc, Nfkd, Nmt, Normalizer, Precompiled,
        Prepend, Replace, Sequence, Strip, StripAccents,
    };
}
