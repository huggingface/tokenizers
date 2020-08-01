pub mod bpe;
pub mod wordpiece;

// Re-export these as decoders
pub use super::pre_tokenizers::byte_level;
pub use super::pre_tokenizers::metaspace;

use serde::{Deserialize, Serialize};

use crate::decoders::bpe::BPEDecoder;
use crate::decoders::wordpiece::WordPiece;
use crate::pre_tokenizers::byte_level::ByteLevel;
use crate::pre_tokenizers::metaspace::Metaspace;
use crate::{Decoder, Result};

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(untagged)]
pub enum DecoderWrapper {
    BPE(BPEDecoder),
    ByteLevel(ByteLevel),
    WordPiece(WordPiece),
    Metaspace(Metaspace),
}

impl Decoder for DecoderWrapper {
    fn decode(&self, tokens: Vec<String>) -> Result<String> {
        match self {
            DecoderWrapper::BPE(bpe) => bpe.decode(tokens),
            DecoderWrapper::ByteLevel(bl) => bl.decode(tokens),
            DecoderWrapper::Metaspace(ms) => ms.decode(tokens),
            DecoderWrapper::WordPiece(wp) => wp.decode(tokens),
        }
    }
}

impl_enum_from!(BPEDecoder, DecoderWrapper, BPE);
impl_enum_from!(ByteLevel, DecoderWrapper, ByteLevel);
impl_enum_from!(Metaspace, DecoderWrapper, Metaspace);
impl_enum_from!(WordPiece, DecoderWrapper, WordPiece);
