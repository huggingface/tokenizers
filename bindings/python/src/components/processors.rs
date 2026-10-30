use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::{Value, json};

use crate::tokenizer::{ComponentKey, ComponentSpec};

use super::{Component, ComponentKind, ComponentType};

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
            _ => PostProcessor::default_component(py, spec),
        }
    }
}

/// A template piece: a sequence, `{"seq": "A"}` or `{"seq": "B"}`, or special token ids,
/// `{"ids": [101]}`. `type_id` defaults to 0.
#[derive(FromPyObject)]
enum Piece {
    Seq {
        #[pyo3(item)]
        seq: String,
        #[pyo3(item, default)]
        type_id: u32,
    },
    Ids {
        #[pyo3(item)]
        ids: Vec<u32>,
        #[pyo3(item, default)]
        type_id: u32,
    },
}

impl Piece {
    fn into_json(self) -> Value {
        match self {
            Self::Seq { seq, type_id } => json!({"seq": seq, "type_id": type_id}),
            Self::Ids { ids, type_id } => json!({"ids": ids, "type_id": type_id}),
        }
    }
}

fn template(pieces: Vec<Bound<'_, PyDict>>) -> PyResult<Value> {
    pieces
        .iter()
        .map(|piece| Ok(piece.extract::<Piece>()?.into_json()))
        .collect()
}

/// Adds special token ids around the sequences, at the places a template marks.
#[pyclass(frozen, extends = PostProcessor, module = "tokenizers.processors")]
#[derive(Default)]
pub struct TemplateProcessing;

impl ComponentType for TemplateProcessing {}

#[pymethods]
impl TemplateProcessing {
    /// `single` defaults to `[{"seq": "A"}]` and `pair` to `[{"seq": "A"}, {"seq": "B", "type_id": 1}]`.
    #[new]
    #[pyo3(signature = (single=None, pair=None))]
    fn new(
        single: Option<Vec<Bound<'_, PyDict>>>,
        pair: Option<Vec<Bound<'_, PyDict>>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let single = single.map_or_else(|| Ok(json!([{"seq": "A", "type_id": 0}])), template)?;
        let pair = pair.map_or_else(
            || Ok(json!([{"seq": "A", "type_id": 0}, {"seq": "B", "type_id": 1}])),
            template,
        )?;
        let Value::Object(json) = json!({
            "type": "TemplateProcessing",
            "single": single,
            "pair": pair,
        }) else {
            unreachable!("json! builds an object from an object literal")
        };
        Ok(Self::initializer(ComponentSpec::new(json)))
    }
}

#[pymodule]
pub mod processors {
    #[pymodule_export]
    use super::{PostProcessor, TemplateProcessing};
}
