#[cxx::bridge(namespace = "huggingface::tokenizers")]
mod ffi {
    extern "C++" {
        include!("tokenizers-cpp/decoders.h");
    }

    #[namespace = "huggingface::tokenizers::ffi"]
    extern "Rust" {
        type Decoder;

        // ignoring ByteLevel::new arguments, as Decoder it doesn't care.
        fn byte_level_decoder() -> Box<Decoder>;

        fn word_piece_decoder(prefix: &str, cleanup: bool) -> Box<Decoder>;

        fn bpe_decoder(suffix: &str) -> Box<Decoder>;

        fn metaspace_decoder(replacement_cp: u32, add_prefix_space: bool) -> Result<Box<Decoder>>;

        fn decode_decoder(decoder: &Decoder, tokens: Vec<String>) -> Result<String>;
    }
}

use derive_more::{Deref, DerefMut};
use tk::{
    decoders::{bpe::BPEDecoder, byte_level::ByteLevel, wordpiece::WordPiece},
    pre_tokenizers::metaspace::Metaspace,
    Decoder as DecoderTrait, DecoderWrapper, Result,
};

use crate::pre_tokenizers::u32_to_char;

#[derive(Deref, DerefMut, Clone)]
pub struct Decoder(pub DecoderWrapper);

impl DecoderTrait for Decoder {
    fn decode(&self, tokens: Vec<String>) -> Result<String> {
        self.0.decode(tokens)
    }
}

fn make_decoder<D: Into<DecoderWrapper>>(decoder: D) -> Box<Decoder> {
    Box::new(Decoder(decoder.into()))
}

fn byte_level_decoder() -> Box<Decoder> {
    make_decoder(ByteLevel::default())
}

fn word_piece_decoder(prefix: &str, cleanup: bool) -> Box<Decoder> {
    make_decoder(WordPiece::new(prefix.to_string(), cleanup))
}

fn bpe_decoder(suffix: &str) -> Box<Decoder> {
    make_decoder(BPEDecoder::new(suffix.to_string()))
}

fn metaspace_decoder(replacement_cp: u32, add_prefix_space: bool) -> Result<Box<Decoder>> {
    Ok(make_decoder(Metaspace::new(
        u32_to_char(replacement_cp, "Replacement")?,
        add_prefix_space,
    )))
}

fn decode_decoder(decoder: &Decoder, tokens: Vec<String>) -> Result<String> {
    decoder.decode(tokens)
}
