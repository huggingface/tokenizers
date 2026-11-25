//! The step that rewrites each piece of text between added tokens.
//!
//! [`Tokenizer::encode`] cuts the added tokens out of the text twice, once before the normalizers run and once after.
//! A piece is the text left between two added tokens.
//! The piece normalizer rewrites each piece on its own, so a prefix it writes never lands in front of an added token.
//!
//! [`Tokenizer::encode`]: super::Tokenizer::encode

use crate::piece_normalizers::{Metaspace, PrefixSpace};

/// Rewrites each piece of text between added tokens, after they are cut.
#[derive(Debug)]
#[non_exhaustive]
pub enum PieceNormalizer {
    Metaspace(Metaspace),
    PrefixSpace(PrefixSpace),
}

impl PieceNormalizer {
    /// Appends the rewritten `piece` to `out`. `opens_sequence` says whether `piece` starts its
    /// sequence.
    pub fn normalize_into(&self, piece: &str, opens_sequence: bool, out: &mut String) {
        match self {
            Self::Metaspace(metaspace) => metaspace.normalize_into(piece, opens_sequence, out),
            Self::PrefixSpace(prefix_space) => prefix_space.normalize_into(piece, out),
        }
    }
}
