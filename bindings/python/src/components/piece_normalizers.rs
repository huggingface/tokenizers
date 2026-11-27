//! Python classes definitions for piece normalizer components.
//! Generated from the definitions in `tk::schemas`.

use super::macros::component_module;

/// Writes the `#[pymethods]` block of each piece normalizer class, adding
/// `Metaspace.from_sentencepiece` to the generated methods.
macro_rules! piece_normalizer_methods {
    (Metaspace { $($generated:tt)* }) => {
        #[pymethods]
        impl Metaspace {
            $($generated)*

            /// Builds the `Metaspace` that rewrites whitespace like SentencePiece, from the
            /// flags in a `.model` file's `normalizer_spec`. The defaults are SentencePiece's.
            #[staticmethod]
            #[pyo3(signature = (
                *,
                add_dummy_prefix = true,
                remove_extra_whitespaces = true,
                escape_whitespaces = true,
            ))]
            fn from_sentencepiece<'py>(
                py: Python<'py>,
                add_dummy_prefix: bool,
                remove_extra_whitespaces: bool,
                escape_whitespaces: bool,
            ) -> PyResult<Bound<'py, Self>> {
                let metaspace = tk::piece_normalizers::Metaspace::from_sentencepiece(
                    add_dummy_prefix,
                    remove_extra_whitespaces,
                    escape_whitespaces,
                );
                let json = tk::piece_normalizer_to_json(Some(
                    &tk::pipeline::PieceNormalizer::Metaspace(metaspace),
                ))
                .map_err($crate::error::err)?;
                let json = serde_json::from_str(&json).expect("to_json writes valid JSON");
                let spec = ComponentSpec::from_literal(json);
                Bound::new(py, PieceNormalizer::initializer(spec, Self))
            }
        }
    };
    ($class:ident { $($generated:tt)* }) => {
        #[pymethods]
        impl $class {
            $($generated)*
        }
    };
}

tk::piece_normalizer_schemas!(component_module! {
    kind: PieceNormalizer,
    module: piece_normalizers,
    python_module: "tokenizers.piece_normalizers",
    tokenizer_field: piece_normalizer,
    class_methods: piece_normalizer_methods,
});
