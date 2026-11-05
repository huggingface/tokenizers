use pyo3::prelude::*;

use crate::components::component_spec::ComponentSpec;
use crate::tokenizer::TokenizerSpec;

use super::macros::component_kind;
use super::{Component, ComponentKind, ConcreteClasses};

component_kind! {
    kind: PostProcessor,
    python_module: "tokenizers.processors",
    tokenizer_field: post_processor,
    sequence_item_key: Some("processors"),
}

impl ConcreteClasses for PostProcessor {
    /// No post-processor has a class of its own yet, so every spec builds the plain kind.
    fn from_spec(py: Python<'_>, spec: ComponentSpec) -> PyResult<Py<Self>> {
        Self::default_component(py, spec)
    }
}

#[pymodule]
pub mod processors {
    #[pymodule_export]
    use super::PostProcessor;
}
