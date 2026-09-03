//! The pipeline encode path, over pyo3.
//!
//! `bindings/python` wraps the pre-v1 engine, which is gone; it no longer builds. The Node
//! bindings (`bindings/node/src/pipeline.rs`) were rebuilt from scratch against the pipeline
//! encode path instead of patched, and this crate is the same restart for Python.
//!
//! Three classes. [`tokenizer::Tokenizer`] wraps [`tk_encode::pipeline::PipelineTokenizer`]:
//! `from_file` builds one, `encode`, `encode_batch` and `decode` are the only things you can do
//! with one, and `padding` is the only thing you can change on one. [`encoding::Encoding`] is
//! what an encode returns, [`padding::Padding`] is how a caller pads without editing the file.
//! Both are read-only.
mod arrays;
mod encoding;
mod error;
mod padding;
mod repr;
mod tokenizer;

/// Tokenizers backed by Rust.
///
/// `Tokenizer.from_file` loads a `tokenizer.json`, `encode` and `encode_batch` turn text into
/// `Encoding`s, `decode` turns ids back into text. `Padding` says how encodings are padded.
#[pyo3::pymodule]
mod tokenizers {
    #[allow(non_upper_case_globals)]
    #[pymodule_export]
    const __version__: &str = env!("CARGO_PKG_VERSION");

    #[pymodule_export]
    use crate::encoding::Encoding;
    #[pymodule_export]
    use crate::padding::Padding;
    #[pymodule_export]
    use crate::tokenizer::Tokenizer;
}
