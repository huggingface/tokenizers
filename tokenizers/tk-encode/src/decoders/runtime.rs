//! The decoder a [`PipelineTokenizer`](crate::pipeline::PipelineTokenizer) actually holds.
//!
//! Same ten decoders the config layer's `DecoderWrapper` had, and the same `Decoder` behaviour —
//! the difference is that this enum carries no serde at all. That wrapper existed for its
//! hand-written `Deserialize` (an untagged legacy fallback over all ten variants), which is exactly
//! what the split moved out and what the strip then deleted: a canonical config is tagged, so
//! `tk-serialize` matches on the tag and builds one of these directly.
//!
//! `Sequence` holds its children inline, as `Vec<DecoderRuntime>`, rather than reusing a `Sequence`
//! *decoder* struct. The config layer's `decoders::sequence::Sequence` was a `Vec<DecoderWrapper>`
//! and needed the wrapper to be parameterised by; this one needs nothing but itself.

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
