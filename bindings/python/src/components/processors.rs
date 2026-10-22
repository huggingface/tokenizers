use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use pyo3::prelude::*;
use serde_json::{Map, Value, json};

use crate::error::err;
use crate::tokenizer::{ComponentKey, ComponentSpec};

use super::{Component, ComponentKind, ComponentSequence, ComponentType};

/// Base class for all post-processors
#[pyclass(frozen, subclass, extends = Component, module = "tokenizers.processors")]
#[derive(Default)]
pub struct PostProcessor;

#[pymethods]
impl PostProcessor {
    /// Rebuilds a component from its `__getstate__`
    #[staticmethod]
    fn _unpickle(py: Python<'_>, state: &[u8]) -> PyResult<Py<Self>> {
        Self::from_state(py, state)
    }
}

impl ComponentKind for PostProcessor {
    const JSON_KEY: ComponentKey = ComponentKey::PostProcessor;

    fn from_spec(py: Python<'_>, spec: Arc<ComponentSpec>) -> PyResult<Py<Self>> {
        match spec.json().get("type").and_then(Value::as_str) {
            Some("TemplateProcessing") => TemplateProcessing::from_spec(py, spec),
            Some("BertProcessing") => BertProcessing::from_spec(py, spec),
            Some("RobertaProcessing") => RobertaProcessing::from_spec(py, spec),
            Some("ByteLevel") => ByteLevel::from_spec(py, spec),
            Some("Sequence") => Sequence::from_spec(py, spec),
            _ => PostProcessor::default_component(py, spec),
        }
    }
}

/// A template, as a string of pieces separated by single spaces or as a list of pieces.
#[derive(FromPyObject)]
enum Template {
    Text(String),
    Pieces(Vec<String>),
}

impl Template {
    fn pieces(&self) -> PyResult<Vec<Value>> {
        match self {
            Self::Text(text) => text.split(' ').map(piece).collect(),
            Self::Pieces(pieces) => pieces.iter().map(|p| piece(p)).collect(),
        }
    }
}

/// Parses a piece as 0.23 does. `$A`, `$B` and `$N` name a sequence, `$N` being sequence A with
/// type id N. Any other text names a special token. A `:N` suffix sets the type id.
fn piece(text: &str) -> PyResult<Value> {
    let error = || err(format!("Cannot build Piece from string \"{text}\""));
    let (id, type_id) = match text.split(':').collect::<Vec<_>>()[..] {
        [id] => (id, None),
        [id, type_id] => (id, Some(type_id.parse::<u32>().map_err(|_| error())?)),
        _ => return Err(error()),
    };
    let (kind, id, implied_type_id) = match id.strip_prefix('$') {
        None => ("SpecialToken", id, 0),
        Some("" | "A" | "a") => ("Sequence", "A", 0),
        Some("B" | "b") => ("Sequence", "B", 0),
        Some(n) => ("Sequence", "A", n.parse().map_err(|_| error())?),
    };
    Ok(json!({kind: {"id": id, "type_id": type_id.unwrap_or(implied_type_id)}}))
}

/// Adds special tokens around the sequences, at the places a template marks.
#[pyclass(frozen, extends = PostProcessor, module = "tokenizers.processors")]
#[derive(Default)]
pub struct TemplateProcessing;

impl ComponentType for TemplateProcessing {}

#[pymethods]
impl TemplateProcessing {
    /// `single` defaults to `"$A"` and `pair` to `"$A $B:1"`.
    #[new]
    #[pyo3(signature = (single=None, pair=None, special_tokens=None))]
    fn new(
        single: Option<Template>,
        pair: Option<Template>,
        special_tokens: Option<Vec<(String, u32)>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let single = single.unwrap_or(Template::Text("$A".to_owned())).pieces()?;
        let pair = pair
            .unwrap_or(Template::Text("$A $B:1".to_owned()))
            .pieces()?;
        let sequences: Vec<_> = pair
            .iter()
            .filter_map(|p| p["Sequence"]["id"].as_str())
            .collect();
        if !(sequences.contains(&"A") && sequences.contains(&"B")) {
            return Err(err("Template for `pair` must use both sequences"));
        }
        // 0.23 keeps the last id given for a token, and writes the tokens sorted by name.
        let special_tokens: BTreeMap<String, u32> = special_tokens.into_iter().flatten().collect();
        let missing: BTreeSet<&str> = single
            .iter()
            .chain(&pair)
            .filter_map(|p| p["SpecialToken"]["id"].as_str())
            .filter(|id| !special_tokens.contains_key(*id))
            .collect();
        if !missing.is_empty() {
            let missing = Vec::from_iter(missing).join(", ");
            return Err(err(format!(
                "Missing SpecialToken(s) with id(s) `{missing}`"
            )));
        }
        let special_tokens: Map<String, Value> = special_tokens
            .into_iter()
            .map(|(token, id)| {
                let entry = json!({"id": token, "ids": [id], "tokens": [token]});
                (token, entry)
            })
            .collect();
        let Value::Object(json) = json!({
            "type": "TemplateProcessing",
            "single": single,
            "pair": pair,
            "special_tokens": special_tokens,
        }) else {
            unreachable!("json! builds an object from an object literal")
        };
        Ok(Self::initializer(ComponentSpec::new(json)))
    }
}

/// BERT's special tokens: `[CLS] A [SEP]`, and `[CLS] A [SEP] B [SEP]` for a pair.
#[pyclass(frozen, extends = PostProcessor, module = "tokenizers.processors")]
#[derive(Default)]
pub struct BertProcessing;

impl ComponentType for BertProcessing {}

#[pymethods]
impl BertProcessing {
    #[new]
    fn new(sep: (String, u32), cls_token: (String, u32)) -> PyClassInitializer<Self> {
        let Value::Object(json) = json!({"type": "BertProcessing", "sep": sep, "cls": cls_token})
        else {
            unreachable!("json! builds an object from an object literal")
        };
        Self::initializer(ComponentSpec::new(json))
    }
}

/// RoBERTa's special tokens: `<s> A </s>`, and `<s> A </s></s> B </s>` for a pair.
#[pyclass(frozen, extends = PostProcessor, module = "tokenizers.processors")]
#[derive(Default)]
pub struct RobertaProcessing;

impl ComponentType for RobertaProcessing {}

#[pymethods]
impl RobertaProcessing {
    #[new]
    #[pyo3(signature = (sep, cls_token, trim_offsets=true, add_prefix_space=true))]
    fn new(
        sep: (String, u32),
        cls_token: (String, u32),
        trim_offsets: bool,
        add_prefix_space: bool,
    ) -> PyClassInitializer<Self> {
        let Value::Object(json) = json!({
            "type": "RobertaProcessing",
            "sep": sep,
            "cls": cls_token,
            "trim_offsets": trim_offsets,
            "add_prefix_space": add_prefix_space,
        }) else {
            unreachable!("json! builds an object from an object literal")
        };
        Self::initializer(ComponentSpec::new(json))
    }
}

/// Removes the leading space that byte-level tokens carry from their offsets.
#[pyclass(frozen, extends = PostProcessor, module = "tokenizers.processors")]
#[derive(Default)]
pub struct ByteLevel;

impl ComponentType for ByteLevel {}

#[pymethods]
impl ByteLevel {
    /// `None` reads as `True`, like in 0.23.
    #[new]
    #[pyo3(signature = (add_prefix_space=None, trim_offsets=None, use_regex=None))]
    fn new(
        add_prefix_space: Option<bool>,
        trim_offsets: Option<bool>,
        use_regex: Option<bool>,
    ) -> PyClassInitializer<Self> {
        let Value::Object(json) = json!({
            "type": "ByteLevel",
            "add_prefix_space": add_prefix_space.unwrap_or(true),
            "trim_offsets": trim_offsets.unwrap_or(true),
            "use_regex": use_regex.unwrap_or(true),
        }) else {
            unreachable!("json! builds an object from an object literal")
        };
        Self::initializer(ComponentSpec::new(json))
    }
}

/// Applies each post-processor in turn.
#[pyclass(frozen, extends = PostProcessor, module = "tokenizers.processors")]
#[derive(Default)]
pub struct Sequence;

impl ComponentType for Sequence {}

impl ComponentSequence for Sequence {
    type Kind = PostProcessor;
    const ITEMS_KEY: &'static str = "processors";
}

#[pymethods]
impl Sequence {
    #[new]
    fn new(processors: Vec<Bound<'_, PostProcessor>>) -> PyClassInitializer<Self> {
        Self::from_items(processors, Self)
    }
}

#[pymodule]
pub mod processors {
    #[pymodule_export]
    use super::{
        BertProcessing, ByteLevel, PostProcessor, RobertaProcessing, Sequence, TemplateProcessing,
    };
}
