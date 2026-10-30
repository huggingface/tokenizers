//! Python bindings for Hugging Face's Tokenizers rust library
//!
//! Encode text to token ids and decode token ids back to text
mod added_token;
mod compiled;
mod encoding;
mod error;
mod input;
mod options;
mod repr;
mod tokenizer;
mod type_hints;

type JsonObject = serde_json::Map<String, serde_json::Value>;

#[pyo3::pymodule]
mod tokenizers {
    #[allow(non_upper_case_globals)]
    #[pymodule_export]
    const __version__: &str = env!("CARGO_PKG_VERSION");

    #[pymodule_export]
    use crate::added_token::AddedToken;
    #[pymodule_export]
    use crate::compiled::CompiledTokenizer;
    #[pymodule_export]
    use crate::encoding::Encoding;
    #[pymodule_export]
    use crate::options::Padding;
    #[pymodule_export]
    use crate::options::Truncation;
    #[pymodule_export]
    use crate::tokenizer::Tokenizer;

    // TODO: bind trainers
}
