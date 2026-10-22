use std::sync::Arc;

use pyo3::prelude::*;
use serde_json::{Value, json};

use crate::error::err;
use crate::regex::Pattern;
use crate::tokenizer::{ComponentKey, ComponentSpec};

use super::{Component, ComponentKind, ComponentSequence, ComponentType, JsonFields};

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
            Some("ByteFallback") => ByteFallback::from_spec(py, spec),
            Some("Fuse") => Fuse::from_spec(py, spec),
            Some("ByteLevel") => ByteLevel::from_spec(py, spec),
            Some("Metaspace") => Metaspace::from_spec(py, spec),
            Some("WordPiece") => WordPiece::from_spec(py, spec),
            Some("BPEDecoder") => BpeDecoder::from_spec(py, spec),
            Some("Replace") => Replace::from_spec(py, spec),
            Some("Strip") => Strip::from_spec(py, spec),
            Some("Sequence") => Sequence::from_spec(py, spec),
            _ => Self::default_component(py, spec),
        }
    }
}

/// Turns each run of `<0xNN>` byte tokens into the text those bytes encode in UTF-8. A run
/// that is not valid UTF-8 becomes one `U+FFFD` per byte.
#[pyclass(frozen, extends = Decoder, module = "tokenizers.decoders")]
#[derive(Default)]
pub struct ByteFallback;

impl ComponentType for ByteFallback {}

#[pymethods]
impl ByteFallback {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        let Value::Object(json) = json!({"type": "ByteFallback"}) else {
            unreachable!("json! builds an object from an object literal")
        };
        Self::initializer(ComponentSpec::new(json))
    }
}

/// Joins every token into one string, with no separator.
#[pyclass(frozen, extends = Decoder, module = "tokenizers.decoders")]
#[derive(Default)]
pub struct Fuse;

impl ComponentType for Fuse {}

#[pymethods]
impl Fuse {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        let Value::Object(json) = json!({"type": "Fuse"}) else {
            unreachable!("json! builds an object from an object literal")
        };
        Self::initializer(ComponentSpec::new(json))
    }
}

/// Maps each character of the GPT-2 byte-to-unicode alphabet back to its byte, and decodes the
/// bytes as UTF-8.
#[pyclass(frozen, extends = Decoder, module = "tokenizers.decoders")]
#[derive(Default)]
pub struct ByteLevel;

impl ComponentType for ByteLevel {}

#[pymethods]
impl ByteLevel {
    // 0.23 ignores these keywords, which transformers passes, and always writes the defaults.
    #[new]
    #[pyo3(signature = (*, add_prefix_space=None, trim_offsets=None, use_regex=None))]
    fn new(
        add_prefix_space: Option<bool>,
        trim_offsets: Option<bool>,
        use_regex: Option<bool>,
    ) -> PyClassInitializer<Self> {
        let _ = (add_prefix_space, trim_offsets, use_regex);
        let Value::Object(json) = json!({
            "type": "ByteLevel",
            "add_prefix_space": true,
            "trim_offsets": true,
            "use_regex": true,
        }) else {
            unreachable!("json! builds an object from an object literal")
        };
        Self::initializer(ComponentSpec::new(json))
    }
}

/// Turns each `replacement` character back into a space. Unless `prepend_scheme` is `"never"`,
/// the first token's `replacement` characters are dropped instead.
#[pyclass(frozen, extends = Decoder, module = "tokenizers.decoders")]
#[derive(Default)]
pub struct Metaspace;

impl ComponentType for Metaspace {}

#[pymethods]
impl Metaspace {
    #[new]
    #[pyo3(signature = (replacement='▁', prepend_scheme="always", split=true))]
    fn new(
        replacement: char,
        prepend_scheme: &str,
        split: bool,
    ) -> PyResult<PyClassInitializer<Self>> {
        if !matches!(prepend_scheme, "always" | "first" | "never") {
            return Err(err(format!(
                "{prepend_scheme} is an unknown variant, should be one of ['first', 'never', 'always']"
            )));
        }
        let Value::Object(json) = json!({
            "type": "Metaspace",
            "replacement": replacement,
            "prepend_scheme": prepend_scheme,
            "split": split,
        }) else {
            unreachable!("json! builds an object from an object literal")
        };
        Ok(Self::initializer(ComponentSpec::new(json)))
    }
}

/// Joins WordPiece tokens, gluing each token that starts with `prefix` to the one before it.
/// `cleanup` also removes the spaces English tokenization puts before punctuation and
/// contractions.
#[pyclass(frozen, extends = Decoder, module = "tokenizers.decoders")]
#[derive(Default)]
pub struct WordPiece;

impl ComponentType for WordPiece {}

#[pymethods]
impl WordPiece {
    #[new]
    #[pyo3(signature = (prefix="##", cleanup=true))]
    fn new(prefix: &str, cleanup: bool) -> PyClassInitializer<Self> {
        let Value::Object(json) =
            json!({"type": "WordPiece", "prefix": prefix, "cleanup": cleanup})
        else {
            unreachable!("json! builds an object from an object literal")
        };
        Self::initializer(ComponentSpec::new(json))
    }

    #[getter]
    fn cleanup(slf: &Bound<'_, Self>) -> PyResult<bool> {
        slf.field("cleanup")
    }

    #[setter]
    fn set_cleanup(slf: &Bound<'_, Self>, cleanup: bool) {
        slf.set_field("cleanup", cleanup);
    }
}

/// Joins BPE tokens, turning each `suffix` that ends a word into a space.
#[pyclass(frozen, extends = Decoder, name = "BPEDecoder", module = "tokenizers.decoders")]
#[derive(Default)]
pub struct BpeDecoder;

impl ComponentType for BpeDecoder {}

#[pymethods]
impl BpeDecoder {
    #[new]
    #[pyo3(signature = (suffix="</w>"))]
    fn new(suffix: &str) -> PyClassInitializer<Self> {
        let Value::Object(json) = json!({"type": "BPEDecoder", "suffix": suffix}) else {
            unreachable!("json! builds an object from an object literal")
        };
        Self::initializer(ComponentSpec::new(json))
    }
}

/// Replaces every match of `pattern` with `content` in each token. A `str` pattern is a literal.
#[pyclass(frozen, extends = Decoder, module = "tokenizers.decoders")]
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

/// Removes up to `left` leading and `right` trailing `content` characters from each token.
#[pyclass(frozen, extends = Decoder, module = "tokenizers.decoders")]
#[derive(Default)]
pub struct Strip;

impl ComponentType for Strip {}

#[pymethods]
impl Strip {
    #[new]
    #[pyo3(signature = (content=' ', left=0, right=0))]
    fn new(content: char, left: usize, right: usize) -> PyClassInitializer<Self> {
        let Value::Object(json) =
            json!({"type": "Strip", "content": content, "start": left, "stop": right})
        else {
            unreachable!("json! builds an object from an object literal")
        };
        Self::initializer(ComponentSpec::new(json))
    }
}

/// Applies each decoder in turn.
#[pyclass(frozen, extends = Decoder, module = "tokenizers.decoders")]
#[derive(Default)]
pub struct Sequence;

impl ComponentType for Sequence {}

impl ComponentSequence for Sequence {
    type Kind = Decoder;
    const ITEMS_KEY: &'static str = "decoders";
}

#[pymethods]
impl Sequence {
    #[new]
    fn new(decoders: Vec<Bound<'_, Decoder>>) -> PyClassInitializer<Self> {
        Self::from_items(decoders, Self)
    }
}

#[pymodule]
pub mod decoders {
    #[pymodule_export]
    use super::{
        BpeDecoder, ByteFallback, ByteLevel, Decoder, Fuse, Metaspace, Replace, Sequence, Strip,
        WordPiece,
    };
}
