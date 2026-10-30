use std::sync::Arc;

use base64::Engine as _;
use pyo3::prelude::*;
use serde_json::{Value, json};

use crate::regex::Pattern;
use crate::tokenizer::{ComponentKey, ComponentSpec};

use super::{Component, ComponentKind, ComponentSequence, ComponentType, JsonFields};

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
            Some("Lowercase") => Lowercase::from_spec(py, spec),
            Some("ByteLevel") => ByteLevel::from_spec(py, spec),
            Some("StripAccents") => StripAccents::from_spec(py, spec),
            Some("NFC") => Nfc::from_spec(py, spec),
            Some("NFD") => Nfd::from_spec(py, spec),
            Some("NFKC") => Nfkc::from_spec(py, spec),
            Some("NFKD") => Nfkd::from_spec(py, spec),
            Some("Nmt") => Nmt::from_spec(py, spec),
            Some("Replace") => Replace::from_spec(py, spec),
            Some("Precompiled") => Precompiled::from_spec(py, spec),
            Some("BertNormalizer") => BertNormalizer::from_spec(py, spec),
            Some("Strip") => Strip::from_spec(py, spec),
            Some("Prepend") => Prepend::from_spec(py, spec),
            Some("MetaspaceNormalizer") => MetaspaceNormalizer::from_spec(py, spec),
            Some("Sequence") => Sequence::from_spec(py, spec),
            _ => Normalizer::default_component(py, spec),
        }
    }
}

/// Lowercases the text, like `str.lower`.
#[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
#[derive(Default)]
pub struct Lowercase;

impl ComponentType for Lowercase {}

#[pymethods]
impl Lowercase {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        let Value::Object(json) = json!({"type": "Lowercase"}) else {
            unreachable!("json! builds an object from an object literal")
        };
        Self::initializer(ComponentSpec::new(json))
    }
}

/// Replaces each byte of the text with its character in the GPT-2 byte-to-unicode mapping.
#[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
#[derive(Default)]
pub struct ByteLevel;

impl ComponentType for ByteLevel {}

#[pymethods]
impl ByteLevel {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        let Value::Object(json) = json!({"type": "ByteLevel"}) else {
            unreachable!("json! builds an object from an object literal")
        };
        Self::initializer(ComponentSpec::new(json))
    }
}

/// Removes combining marks. Use it after `NFD` or `NFKD` to strip accents.
#[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
#[derive(Default)]
pub struct StripAccents;

impl ComponentType for StripAccents {}

#[pymethods]
impl StripAccents {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        let Value::Object(json) = json!({"type": "StripAccents"}) else {
            unreachable!("json! builds an object from an object literal")
        };
        Self::initializer(ComponentSpec::new(json))
    }
}

/// Unicode NFC normalization.
#[pyclass(frozen, extends = Normalizer, name = "NFC", module = "tokenizers.normalizers")]
#[derive(Default)]
pub struct Nfc;

impl ComponentType for Nfc {}

#[pymethods]
impl Nfc {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        let Value::Object(json) = json!({"type": "NFC"}) else {
            unreachable!("json! builds an object from an object literal")
        };
        Self::initializer(ComponentSpec::new(json))
    }
}

/// Unicode NFD normalization.
#[pyclass(frozen, extends = Normalizer, name = "NFD", module = "tokenizers.normalizers")]
#[derive(Default)]
pub struct Nfd;

impl ComponentType for Nfd {}

#[pymethods]
impl Nfd {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        let Value::Object(json) = json!({"type": "NFD"}) else {
            unreachable!("json! builds an object from an object literal")
        };
        Self::initializer(ComponentSpec::new(json))
    }
}

/// Unicode NFKC normalization.
#[pyclass(frozen, extends = Normalizer, name = "NFKC", module = "tokenizers.normalizers")]
#[derive(Default)]
pub struct Nfkc;

impl ComponentType for Nfkc {}

#[pymethods]
impl Nfkc {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        let Value::Object(json) = json!({"type": "NFKC"}) else {
            unreachable!("json! builds an object from an object literal")
        };
        Self::initializer(ComponentSpec::new(json))
    }
}

/// Unicode NFKD normalization.
#[pyclass(frozen, extends = Normalizer, name = "NFKD", module = "tokenizers.normalizers")]
#[derive(Default)]
pub struct Nfkd;

impl ComponentType for Nfkd {}

#[pymethods]
impl Nfkd {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        let Value::Object(json) = json!({"type": "NFKD"}) else {
            unreachable!("json! builds an object from an object literal")
        };
        Self::initializer(ComponentSpec::new(json))
    }
}

/// Removes control characters and replaces line breaks, tabs and some invisible characters with a
/// space, like the Google NMT pipeline.
#[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
#[derive(Default)]
pub struct Nmt;

impl ComponentType for Nmt {}

#[pymethods]
impl Nmt {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        let Value::Object(json) = json!({"type": "Nmt"}) else {
            unreachable!("json! builds an object from an object literal")
        };
        Self::initializer(ComponentSpec::new(json))
    }
}

/// Replaces every match of `pattern` with `content`. A `str` pattern is a literal.
#[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
#[derive(Default)]
pub struct Replace;

impl ComponentType for Replace {}

#[pymethods]
impl Replace {
    #[new]
    fn new(pattern: Pattern, content: String) -> PyClassInitializer<Self> {
        let Value::Object(json) =
            json!({"type": "Replace", "pattern": Value::from(pattern), "content": content})
        else {
            unreachable!("json! builds an object from an object literal")
        };
        Self::initializer(ComponentSpec::new(json))
    }
}

/// SentencePiece's normalization rules, compiled into a charsmap (T5, ALBERT, XLM-R...).
#[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
#[derive(Default)]
pub struct Precompiled;

impl ComponentType for Precompiled {}

#[pymethods]
impl Precompiled {
    #[new]
    fn new(precompiled_charsmap: &[u8]) -> PyClassInitializer<Self> {
        let charsmap = base64::engine::general_purpose::STANDARD.encode(precompiled_charsmap);
        let Value::Object(json) = json!({"type": "Precompiled", "precompiled_charsmap": charsmap})
        else {
            unreachable!("json! builds an object from an object literal")
        };
        Self::initializer(ComponentSpec::new(json))
    }
}

/// BERT's normalization: control characters removed, spaces around CJK characters, accents
/// stripped and the text lowercased, each optional.
#[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
#[derive(Default)]
pub struct BertNormalizer;

impl ComponentType for BertNormalizer {}

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
        let Value::Object(json) = json!({
            "type": "BertNormalizer",
            "clean_text": clean_text,
            "handle_chinese_chars": handle_chinese_chars,
            "strip_accents": strip_accents,
            "lowercase": lowercase,
        }) else {
            unreachable!("json! builds an object from an object literal")
        };
        Self::initializer(ComponentSpec::new(json))
    }

    #[getter]
    fn strip_accents(slf: &Bound<'_, Self>) -> PyResult<Option<bool>> {
        slf.field("strip_accents")
    }
}

/// Removes whitespace from the start and the end of the text.
#[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
#[derive(Default)]
pub struct Strip;

impl ComponentType for Strip {}

#[pymethods]
impl Strip {
    #[new]
    #[pyo3(signature = (left=true, right=true))]
    fn new(left: bool, right: bool) -> PyClassInitializer<Self> {
        let Value::Object(json) =
            json!({"type": "Strip", "strip_left": left, "strip_right": right})
        else {
            unreachable!("json! builds an object from an object literal")
        };
        Self::initializer(ComponentSpec::new(json))
    }
}

/// Adds `prepend` at the start of the text, unless the text is empty.
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

/// SentencePiece's whitespace handling: replaces spaces with `replacement`. `prepend` says when
/// `replacement` also goes in front of the text: `"always"`, `"first"` (only at the start of the
/// input) or `"never"`. `drop_whitespace` drops runs of whitespace instead of replacing them.
#[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
#[derive(Default)]
pub struct MetaspaceNormalizer;

impl ComponentType for MetaspaceNormalizer {}

#[pymethods]
impl MetaspaceNormalizer {
    #[new]
    #[pyo3(signature = (replacement='▁', prepend="always", drop_whitespace=false))]
    fn new(replacement: char, prepend: &str, drop_whitespace: bool) -> PyClassInitializer<Self> {
        let Value::Object(json) = json!({
            "type": "MetaspaceNormalizer",
            "replacement": replacement,
            "prepend": prepend,
            "drop_whitespace": drop_whitespace,
        }) else {
            unreachable!("json! builds an object from an object literal")
        };
        Self::initializer(ComponentSpec::new(json))
    }
}

/// Applies each normalizer in turn. Items read from it are copies: assign `seq[i] = item` to change one.
#[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
#[derive(Default)]
pub struct Sequence;

impl ComponentType for Sequence {}

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
        BertNormalizer, ByteLevel, Lowercase, MetaspaceNormalizer, Nfc, Nfd, Nfkc, Nfkd, Nmt,
        Normalizer, Precompiled, Prepend, Replace, Sequence, Strip, StripAccents,
    };
}
