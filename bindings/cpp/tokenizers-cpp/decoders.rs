#[cxx::bridge(namespace = "huggingface::tokenizers")]
mod ffi {
    extern "C++" {
        include!("tokenizers-cpp/decoders.h");
    }

    #[namespace = "huggingface::tokenizers::ffi"]
    extern "Rust" {
        type Decoder;

        fn byte_level_decoder(add_prefix_space: bool, trim_offsets: bool) -> Box<Decoder>;
        fn word_piece_decoder(prefix: &str, cleanup: bool) -> Box<Decoder>;

        fn decode_decoder(decoder: &Decoder, tokens: Vec<String>) -> Result<String>;
    }
}

use derive_more::{Deref, DerefMut};
use tk::{
    decoders::{byte_level::ByteLevel, wordpiece::WordPiece},
    Decoder as DecoderTrait, Result,
};

#[derive(Deref, DerefMut, Clone)]
pub struct Decoder(pub tk::DecoderWrapper);

impl DecoderTrait for Decoder {
    fn decode(&self, tokens: Vec<String>) -> Result<String> {
        self.0.decode(tokens)
    }
}

fn byte_level_decoder(add_prefix_space: bool, trim_offsets: bool) -> Box<Decoder> {
    Box::new(Decoder(
        ByteLevel::new(add_prefix_space, trim_offsets).into(),
    ))
}

fn word_piece_decoder(prefix: &str, cleanup: bool) -> Box<Decoder> {
    Box::new(Decoder(WordPiece::new(prefix.to_string(), cleanup).into()))
}

fn decode_decoder(decoder: &Decoder, tokens: Vec<String>) -> Result<String> {
    decoder.decode(tokens)
}
