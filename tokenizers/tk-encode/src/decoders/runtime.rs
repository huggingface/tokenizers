//! The decoder a [`PipelineTokenizer`](crate::pipeline::PipelineTokenizer) actually holds.
//!
//! Same ten decoders as the config layer's `DecoderWrapper`, and the same `Decoder` behaviour —
//! the difference is that this enum carries no serde at all. `DecoderWrapper` cannot live here
//! because its `Deserialize` is hand-written BC (an untagged legacy fallback over all ten
//! variants), and that is exactly what the split moves out; a foreign crate cannot add a
//! `Deserialize` impl to a type defined here, so the config layer owns its own enum and lowers it
//! into this one.
//!
//! `Sequence` holds its children inline rather than reusing a `Sequence` *decoder* struct: the
//! config layer's `decoders::sequence::Sequence` is a `Vec<DecoderWrapper>`, so it belongs on that
//! side of the split with the wrapper it is parameterised by.

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
