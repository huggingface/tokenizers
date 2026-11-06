use pyo3::prelude::*;

use super::macros::component_kind;

// A post-processor `Sequence` never reaches a spec: loading a file collapses it to one item.
component_kind! {
    kind: PostProcessor,
    python_module: "tokenizers.processors",
    tokenizer_field: post_processor,
    items_key: None,
    classes: {},
}

#[pymodule]
pub mod processors {
    #[pymodule_export]
    use super::PostProcessor;
}
