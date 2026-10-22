use std::sync::Arc;

use pyo3::prelude::*;
use serde_json::{Value, json};

use crate::error::err;
use crate::regex::Pattern;
use crate::spec::ComponentKey;

use super::{Component, ComponentKind, ComponentSequence, ComponentSpec, JsonFields};

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
            Some("ByteLevel") => Self::subclass_from_spec(py, spec, ByteLevel),
            Some("Metaspace") => Self::subclass_from_spec(py, spec, Metaspace),
            Some("WordPiece") => Self::subclass_from_spec(py, spec, WordPiece),
            Some("BPEDecoder") => Self::subclass_from_spec(py, spec, BpeDecoder),
            Some("Replace") => Self::subclass_from_spec(py, spec, Replace),
            Some("Strip") => Self::subclass_from_spec(py, spec, Strip),
            Some("Sequence") => Self::subclass_from_spec(py, spec, Sequence),
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
        Decoder::init(json!({"type": "ByteFallback"}), Self)
    }
}

/// Joins every token into one string, with no separator.
#[pyclass(frozen, extends = Decoder, module = "tokenizers.decoders")]
pub struct Fuse;

#[pymethods]
impl Fuse {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        Decoder::init(json!({"type": "Fuse"}), Self)
    }
}

/// Maps each character of the GPT-2 byte-to-unicode alphabet back to its byte, and decodes the
/// bytes as UTF-8.
#[pyclass(frozen, extends = Decoder, module = "tokenizers.decoders")]
pub struct ByteLevel;

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
        let json = json!({
            "type": "ByteLevel",
            "add_prefix_space": true,
            "trim_offsets": true,
            "use_regex": true,
        });
        Decoder::init(json, Self)
    }
}

/// Turns each `replacement` character back into a space. Unless `prepend_scheme` is `"never"`,
/// the first token's `replacement` characters are dropped instead.
#[pyclass(frozen, extends = Decoder, module = "tokenizers.decoders")]
pub struct Metaspace;

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
        let json = json!({
            "type": "Metaspace",
            "replacement": replacement,
            "prepend_scheme": prepend_scheme,
            "split": split,
        });
        Ok(Decoder::init(json, Self))
    }
}

/// Joins WordPiece tokens, gluing each token that starts with `prefix` to the one before it.
/// `cleanup` also removes the spaces English tokenization puts before punctuation and
/// contractions.
#[pyclass(frozen, extends = Decoder, module = "tokenizers.decoders")]
pub struct WordPiece;

#[pymethods]
impl WordPiece {
    #[new]
    #[pyo3(signature = (prefix="##", cleanup=true))]
    fn new(prefix: &str, cleanup: bool) -> PyClassInitializer<Self> {
        let json = json!({"type": "WordPiece", "prefix": prefix, "cleanup": cleanup});
        Decoder::init(json, Self)
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
pub struct BpeDecoder;

#[pymethods]
impl BpeDecoder {
    #[new]
    #[pyo3(signature = (suffix="</w>"))]
    fn new(suffix: &str) -> PyClassInitializer<Self> {
        Decoder::init(json!({"type": "BPEDecoder", "suffix": suffix}), Self)
    }
}

/// Replaces every match of `pattern` with `content` in each token. A `str` pattern is a literal.
#[pyclass(frozen, extends = Decoder, module = "tokenizers.decoders")]
pub struct Replace;

#[pymethods]
impl Replace {
    #[new]
    fn new(pattern: Pattern, content: String) -> PyClassInitializer<Self> {
        let json = json!({"type": "Replace", "pattern": Value::from(pattern), "content": content});
        Decoder::init(json, Self)
    }
}

/// Removes up to `left` leading and `right` trailing `content` characters from each token.
#[pyclass(frozen, extends = Decoder, module = "tokenizers.decoders")]
pub struct Strip;

#[pymethods]
impl Strip {
    #[new]
    #[pyo3(signature = (content=' ', left=0, right=0))]
    fn new(content: char, left: usize, right: usize) -> PyClassInitializer<Self> {
        let json = json!({"type": "Strip", "content": content, "start": left, "stop": right});
        Decoder::init(json, Self)
    }
}

/// Applies each decoder in turn.
#[pyclass(frozen, extends = Decoder, module = "tokenizers.decoders")]
pub struct Sequence;

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
