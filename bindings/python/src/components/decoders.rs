use pyo3::prelude::*;

use crate::components::component_spec::ComponentSpec;
use crate::tokenizer::TokenizerSpec;

use super::component_sequence::ComponentSequence;
use super::fields::{arg, value};
use super::macros::{component_classes, sequence_class};
use super::{Component, ComponentKind, JsonFieldsExt, check_choice};

sequence_class! {
    kind: Decoder,
    python_module: "tokenizers.decoders",
}

tk::decoder_schemas!(component_classes! {
    kind: Decoder,
    module: decoders,
    python_module: "tokenizers.decoders",
});

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
    const SEQUENCE_ITEM_KEY: Option<&'static str> = Some("decoders");

    fn tokenizer_field_mut(spec: &mut TokenizerSpec) -> &mut Option<ComponentSpec> {
        &mut spec.decoder
    }

    fn from_spec(py: Python<'_>, spec: ComponentSpec) -> PyResult<Py<Self>> {
        match &spec {
            ComponentSpec::Sequence(_) => Self::concrete_component(py, spec, Sequence),
            ComponentSpec::Single(single) => {
                let json_type = single.json_type();
                from_json_type(py, spec, json_type.as_deref())
            }
        }
    }
}
