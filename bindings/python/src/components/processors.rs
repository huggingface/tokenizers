use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::{Value, json};

use crate::components::component_spec::ComponentSpec;

use super::ComponentKind;
use super::macros::component_kind;

// A post-processor `Sequence` never reaches a spec: loading a file collapses it to one item.
component_kind! {
    kind: PostProcessor,
    python_module: "tokenizers.processors",
    tokenizer_field: post_processor,
    items_key: None,
    classes: { "TemplateProcessing" => TemplateProcessing },
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
pub struct TemplateProcessing;

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
        let spec = ComponentSpec::from_literal(json!({
            "type": "TemplateProcessing",
            "single": single,
            "pair": pair,
        }));
        Ok(PostProcessor::initializer(spec, Self))
    }
}

#[pymodule]
pub mod processors {
    #[pymodule_export]
    use super::{PostProcessor, TemplateProcessing};
}
