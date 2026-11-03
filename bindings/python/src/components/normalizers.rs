use base64::Engine as _;
use pyo3::prelude::*;
use serde_json::{Value, json};

use crate::components::component_spec::ComponentSpec;
use crate::regex::Pattern;
use crate::tokenizer::TokenizerSpec;

use super::sequence_component::sequence_component;
use super::stateless_component::stateless_component;
use super::{Component, ComponentKind, JsonFieldsExt};

const SEQUENCE_ITEMS_KEY: &str = "normalizers";

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
    const ITEMS_KEY: Option<&'static str> = Some(SEQUENCE_ITEMS_KEY);

    fn tokenizer_field_mut(spec: &mut TokenizerSpec) -> &mut Option<ComponentSpec> {
        &mut spec.normalizer
    }

    fn from_spec(py: Python<'_>, spec: ComponentSpec) -> PyResult<Py<Self>> {
        match &spec {
            ComponentSpec::Sequence(_) => Self::concrete_component(py, spec, Sequence),
            ComponentSpec::Single(single) => match single.json_type().as_deref() {
                Some("Lowercase") => Self::concrete_component(py, spec, Lowercase),
                Some("ByteLevel") => Self::concrete_component(py, spec, ByteLevel),
                Some("StripAccents") => Self::concrete_component(py, spec, StripAccents),
                Some("NFC") => Self::concrete_component(py, spec, Nfc),
                Some("NFD") => Self::concrete_component(py, spec, Nfd),
                Some("NFKC") => Self::concrete_component(py, spec, Nfkc),
                Some("NFKD") => Self::concrete_component(py, spec, Nfkd),
                Some("Nmt") => Self::concrete_component(py, spec, Nmt),
                Some("Replace") => Self::concrete_component(py, spec, Replace),
                Some("Precompiled") => Self::concrete_component(py, spec, Precompiled),
                Some("BertNormalizer") => Self::concrete_component(py, spec, BertNormalizer),
                Some("Strip") => Self::concrete_component(py, spec, Strip),
                Some("Prepend") => Self::concrete_component(py, spec, Prepend),
                Some("MetaspaceNormalizer") => {
                    Self::concrete_component(py, spec, MetaspaceNormalizer)
                }
                _ => Self::default_component(py, spec),
            },
        }
    }
}

sequence_component! {
    /// Applies each normalizer in turn.
    #[pyclass(extends = Normalizer, module = "tokenizers.normalizers")]
    pub struct Sequence;
    items_key = SEQUENCE_ITEMS_KEY;
}

stateless_component! {
    /// Lowercases the text, like `str.lower`.
    #[pyclass(extends = Normalizer, name = "Lowercase", module = "tokenizers.normalizers")]
    pub struct Lowercase;
}

stateless_component! {
    /// Replaces each byte of the text with its character in the GPT-2 byte-to-unicode mapping.
    #[pyclass(extends = Normalizer, name = "ByteLevel", module = "tokenizers.normalizers")]
    pub struct ByteLevel;
}

stateless_component! {
    /// Removes combining marks. Use it after `NFD` or `NFKD` to strip accents.
    #[pyclass(extends = Normalizer, name = "StripAccents", module = "tokenizers.normalizers")]
    pub struct StripAccents;
}

stateless_component! {
    /// Unicode NFC normalization.
    #[pyclass(extends = Normalizer, name = "NFC", module = "tokenizers.normalizers")]
    pub struct Nfc;
}

stateless_component! {
    /// Unicode NFD normalization.
    #[pyclass(extends = Normalizer, name = "NFD", module = "tokenizers.normalizers")]
    pub struct Nfd;
}

stateless_component! {
    /// Unicode NFKC normalization.
    #[pyclass(extends = Normalizer, name = "NFKC", module = "tokenizers.normalizers")]
    pub struct Nfkc;
}

stateless_component! {
    /// Unicode NFKD normalization.
    #[pyclass(extends = Normalizer, name = "NFKD", module = "tokenizers.normalizers")]
    pub struct Nfkd;
}

stateless_component! {
    /// Removes control characters and replaces line breaks, tabs and some invisible characters
    /// with a space, like the Google NMT pipeline.
    #[pyclass(extends = Normalizer, name = "Nmt", module = "tokenizers.normalizers")]
    pub struct Nmt;
}

/// Replaces every match of `pattern` with `content`. A `str` pattern is a literal.
#[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
pub struct Replace;

#[pymethods]
impl Replace {
    #[new]
    fn new(pattern: Pattern, content: String) -> PyClassInitializer<Self> {
        Normalizer::initializer(
            ComponentSpec::from_literal(
                json!({"type": "Replace", "pattern": Value::from(pattern), "content": content}),
            ),
            Self,
        )
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
        Normalizer::initializer(
            ComponentSpec::from_literal(
                json!({"type": "Precompiled", "precompiled_charsmap": charsmap}),
            ),
            Self,
        )
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
        Normalizer::initializer(
            ComponentSpec::from_literal(json!({
                "type": "BertNormalizer",
                "clean_text": clean_text,
                "handle_chinese_chars": handle_chinese_chars,
                "strip_accents": strip_accents,
                "lowercase": lowercase,
            })),
            Self,
        )
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
        Normalizer::initializer(
            ComponentSpec::from_literal(
                json!({"type": "Strip", "strip_left": left, "strip_right": right}),
            ),
            Self,
        )
    }
}

/// Adds `prepend` at the start of the text, unless the text is empty.
#[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
pub struct Prepend;

#[pymethods]
impl Prepend {
    #[new]
    fn new(prepend: String) -> PyClassInitializer<Self> {
        Normalizer::initializer(
            ComponentSpec::from_literal(json!({"type": "Prepend", "prepend": prepend})),
            Self,
        )
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
pub struct MetaspaceNormalizer;

#[pymethods]
impl MetaspaceNormalizer {
    #[new]
    #[pyo3(signature = (replacement='▁', prepend="always", drop_whitespace=false))]
    fn new(replacement: char, prepend: &str, drop_whitespace: bool) -> PyClassInitializer<Self> {
        Normalizer::initializer(
            ComponentSpec::from_literal(json!({
                "type": "MetaspaceNormalizer",
                "replacement": replacement,
                "prepend": prepend,
                "drop_whitespace": drop_whitespace,
            })),
            Self,
        )
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
