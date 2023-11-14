pub mod bpe;
pub mod byte_fallback;
pub mod ctc;
pub mod fuse;
pub mod sequence;
pub mod strip;
pub mod wordpiece;

// Re-export these as decoders
pub use super::pre_tokenizers::byte_level;
pub use super::pre_tokenizers::metaspace;

use serde::{Deserialize, Serialize};

use crate::decoders::bpe::BPEDecoder;
use crate::decoders::byte_fallback::ByteFallback;
use crate::decoders::ctc::CTC;
use crate::decoders::fuse::Fuse;
use crate::decoders::sequence::Sequence;
use crate::decoders::strip::Strip;
use crate::decoders::wordpiece::WordPiece;
use crate::normalizers::replace::Replace;
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
    Sequence(Sequence),
    Replace(Replace),
    Fuse(Fuse),
    Strip(Strip),
    ByteFallback(ByteFallback),
}

impl Decoder for DecoderWrapper {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        match self {
            Self::BPE(bpe) => bpe.decode_chain(tokens),
            Self::ByteLevel(bl) => bl.decode_chain(tokens),
            Self::Metaspace(ms) => ms.decode_chain(tokens),
            Self::WordPiece(wp) => wp.decode_chain(tokens),
            Self::CTC(ctc) => ctc.decode_chain(tokens),
            Self::Sequence(seq) => seq.decode_chain(tokens),
            Self::Replace(seq) => seq.decode_chain(tokens),
            Self::ByteFallback(bf) => bf.decode_chain(tokens),
            Self::Strip(bf) => bf.decode_chain(tokens),
            Self::Fuse(bf) => bf.decode_chain(tokens),
        }
    }
}

impl_enum_from!(BPEDecoder, DecoderWrapper, BPE);
impl_enum_from!(ByteLevel, DecoderWrapper, ByteLevel);
impl_enum_from!(ByteFallback, DecoderWrapper, ByteFallback);
impl_enum_from!(Fuse, DecoderWrapper, Fuse);
impl_enum_from!(Strip, DecoderWrapper, Strip);
impl_enum_from!(Metaspace, DecoderWrapper, Metaspace);
impl_enum_from!(WordPiece, DecoderWrapper, WordPiece);
impl_enum_from!(CTC, DecoderWrapper, CTC);
impl_enum_from!(Sequence, DecoderWrapper, Sequence);
impl_enum_from!(Replace, DecoderWrapper, Replace);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_serialization() {
        let json = r#"{"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Metaspace","replacement":"▁","add_prefix_space":true,"prepend_scheme":"always"}]}"#;
        let decoder: DecoderWrapper = serde_json::from_str(json).unwrap();
        let serialized = serde_json::to_string(&decoder).unwrap();
        assert_eq!(serialized, json);
    }
    #[test]
    fn decoder_serialization_other_no_arg() {
        let json = r#"{"type":"Sequence","decoders":[{"type":"Fuse"},{"type":"Metaspace","replacement":"▁","add_prefix_space":true,"prepend_scheme":"always"}]}"#;
        let decoder: DecoderWrapper = serde_json::from_str(json).unwrap();
        let serialized = serde_json::to_string(&decoder).unwrap();
        assert_eq!(serialized, json);
    }

    #[test]
    fn decoder_serialization_no_decode() {
        let json = r#"{"type":"Sequence","decoders":[{},{"type":"Metaspace","replacement":"▁","add_prefix_space":true,"prepend_scheme":"always"}]}"#;
        assert!(serde_json::from_str::<DecoderWrapper>(json).is_err());
    }
}
