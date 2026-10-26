//! Python bindings for Hugging Face's Tokenizers rust library
//!
//! Encode text to token ids and decode token ids back to text
mod compiled;
mod encoding;
mod error;
mod input;
mod options;
mod repr;
mod spec;
mod type_hints;

#[pyo3::pymodule]
mod tokenizers {
    #[allow(non_upper_case_globals)]
    #[pymodule_export]
    const __version__: &str = env!("CARGO_PKG_VERSION");

    #[pymodule_export]
    use crate::compiled::CompiledTokenizer;
    #[pymodule_export]
    use crate::encoding::Encoding;
    #[pymodule_export]
    use crate::options::Padding;
    #[pymodule_export]
    use crate::options::Truncation;
    #[pymodule_export]
    use crate::spec::Tokenizer;

    // TODO: bind trainers
}
