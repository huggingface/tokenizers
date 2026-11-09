//! Python classes definitions for normalizer components.
//! Generated from the definitions in `tk::schemas`.

use super::macros::component_module;

tk::normalizer_schemas!(component_module! {
    kind: Normalizer,
    module: normalizers,
    python_module: "tokenizers.normalizers",
    tokenizer_field: normalizer,
    items_key: "normalizers",
});

#[pyo3::pymethods]
impl ByteLevel {
    /// Returns the 256 characters used to represent bytes by this normalizer.
    ///
    /// Pass this alphabet as `initial_alphabet` when training a byte-level BPE
    /// tokenizer so every byte remains representable, including unseen bytes.
    #[staticmethod]
    fn alphabet() -> Vec<String> {
        tk::pre_tokenizers::byte_level::ByteLevel::alphabet()
            .into_iter()
            .map(|character| character.to_string())
            .collect()
    }
}
