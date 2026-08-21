//! The decoder a [`PipelineTokenizer`](crate::pipeline::PipelineTokenizer) actually holds.
use crate::decoders::bpe::BPEDecoder;
use crate::decoders::byte_fallback::ByteFallback;
use crate::decoders::byte_level::ByteLevelDecoder;
use crate::decoders::ctc::CTC;
use crate::decoders::fuse::Fuse;
use crate::decoders::metaspace::MetaspaceDecoder;
use crate::decoders::replace::ReplaceDecoder;
use crate::decoders::strip::Strip;
use crate::decoders::wordpiece::WordPiece;
use crate::{Decoder, Result};

#[derive(Clone, Debug)]
pub enum DecoderRuntime {
    BPE(BPEDecoder),
    ByteLevel(ByteLevelDecoder),
    WordPiece(WordPiece),
    Metaspace(MetaspaceDecoder),
    CTC(CTC),
    Sequence(Vec<DecoderRuntime>),
    Replace(ReplaceDecoder),
    Fuse(Fuse),
    Strip(Strip),
    ByteFallback(ByteFallback),
}

impl Decoder for DecoderRuntime {
    fn decode_chain(&self, mut tokens: Vec<String>) -> Result<Vec<String>> {
        match self {
            Self::BPE(bpe) => bpe.decode_chain(tokens),
            Self::ByteLevel(bl) => bl.decode_chain(tokens),
            Self::Metaspace(ms) => ms.decode_chain(tokens),
            Self::WordPiece(wp) => wp.decode_chain(tokens),
            Self::CTC(ctc) => ctc.decode_chain(tokens),
            Self::Sequence(members) => {
                for member in members {
                    tokens = member.decode_chain(tokens)?;
                }
                Ok(tokens)
            }
            Self::Replace(rp) => rp.decode_chain(tokens),
            Self::ByteFallback(bf) => bf.decode_chain(tokens),
            Self::Strip(st) => st.decode_chain(tokens),
            Self::Fuse(fs) => fs.decode_chain(tokens),
        }
    }
}
