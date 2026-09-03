//! Python bindings for Hugging Face's Tokenizers rust library
//! 
//! Encode text to token ids and decode token ids back to text
mod type_hints;
mod encoding;
mod error;
mod padding;
mod repr;
mod tokenizer;

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
