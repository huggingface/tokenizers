use pyo3::prelude::*;

use crate::components::component_spec::ComponentSpec;

use super::macros::component_kind;
use super::{ComponentKind, FromSpec};

// A post-processor `Sequence` never reaches a spec: loading a file collapses it to one item.
component_kind! {
    kind: PostProcessor,
    python_module: "tokenizers.processors",
    tokenizer_field: post_processor,
    items_key: None,
}

impl FromSpec for PostProcessor {
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
