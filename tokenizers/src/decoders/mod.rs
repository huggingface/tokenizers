pub mod bpe;
pub mod ctc;
pub mod wordpiece;

// Re-export these as decoders
pub use super::pre_tokenizers::byte_level;
pub use super::pre_tokenizers::metaspace;

use serde::{Deserialize, Serialize};

use crate::decoders::bpe::BPEDecoder;
use crate::decoders::ctc::CTC;
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
    CTC(CTC),
}

impl Decoder for DecoderWrapper {
    fn decode(&self, tokens: Vec<String>) -> Result<String> {
        match self {
            Self::BPE(bpe) => bpe.decode(tokens),
            Self::ByteLevel(bl) => bl.decode(tokens),
            Self::Metaspace(ms) => ms.decode(tokens),
            Self::WordPiece(wp) => wp.decode(tokens),
            Self::CTC(ctc) => ctc.decode(tokens),
        }
    }
}

impl_enum_from!(BPEDecoder, DecoderWrapper, BPE);
impl_enum_from!(ByteLevel, DecoderWrapper, ByteLevel);
impl_enum_from!(Metaspace, DecoderWrapper, Metaspace);
impl_enum_from!(WordPiece, DecoderWrapper, WordPiece);
impl_enum_from!(CTC, DecoderWrapper, CTC);
