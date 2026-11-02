//! Python bindings for Hugging Face's Tokenizers rust library
//!
//! Encode text to token ids and decode token ids back to text
mod compiled;
mod components;
mod encoding;
mod error;
mod input;
mod options;
mod regex;
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
    use crate::compiled::CompiledTokenizer;
    #[pymodule_export]
    use crate::components::Component;
    #[pymodule_export]
    use crate::encoding::Encoding;
    #[pymodule_export]
    use crate::options::Padding;
    #[pymodule_export]
    use crate::options::Truncation;
    #[pymodule_export]
    use crate::regex::Regex;
    #[pymodule_export]
    use crate::tokenizer::Tokenizer;

    #[pymodule_export]
    use crate::components::decoders::decoders;
    #[pymodule_export]
    use crate::components::models::models;
    #[pymodule_export]
    use crate::components::normalizers::normalizers;
    #[pymodule_export]
    use crate::components::pre_tokenizers::pre_tokenizers;
    #[pymodule_export]
    use crate::components::processors::processors;

    // TODO: bind trainers
}
