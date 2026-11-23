//! Python classes definitions for piece normalizer components.
//! Generated from the definitions in `tk::schemas`.

use super::macros::component_module;

tk::piece_normalizer_schemas!(component_module! {
    kind: PieceNormalizer,
    module: piece_normalizers,
    python_module: "tokenizers.piece_normalizers",
    tokenizer_field: piece_normalizer,
});
