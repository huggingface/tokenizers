#[cxx::bridge(namespace = "huggingface::tokenizers")]
mod ffi {
    extern "C++" {
        include!("tokenizers-cpp/decoders.h");
    }

    #[namespace = "huggingface::tokenizers::ffi"]
    extern "Rust" {
        type Decoder;
        type ByteLevelDecoder;

        fn byte_level_decoder(add_prefix_space: bool, trim_offsets: bool) -> Box<ByteLevelDecoder>;
        fn byte_level_decoder_wrapper(add_prefix_space: bool, trim_offsets: bool) -> Box<Decoder>;

        fn decode_byte_level(decoder: &ByteLevelDecoder, tokens: Vec<String>) -> Result<String>;
    }
}

use derive_more::{Deref, DerefMut};
use tk::{decoders::byte_level::ByteLevel as TkByteLevelDecoder, Decoder as DecoderTrait, Result};

#[derive(Deref, DerefMut, Clone)]
pub struct Decoder(pub tk::DecoderWrapper);

impl DecoderTrait for Decoder {
    fn decode(&self, tokens: Vec<String>) -> Result<String> {
        self.0.decode(tokens)
    }
}

#[derive(Deref, DerefMut)]
struct ByteLevelDecoder(TkByteLevelDecoder);

fn byte_level_decoder(add_prefix_space: bool, trim_offsets: bool) -> Box<ByteLevelDecoder> {
    Box::new(ByteLevelDecoder(TkByteLevelDecoder::new(
        add_prefix_space,
        trim_offsets,
    )))
}

fn byte_level_decoder_wrapper(add_prefix_space: bool, trim_offsets: bool) -> Box<Decoder> {
    Box::new(Decoder(TkByteLevelDecoder::new(
        add_prefix_space,
        trim_offsets,
    ).into()))
}

fn decode_byte_level(decoder: &ByteLevelDecoder, tokens: Vec<String>) -> Result<String> {
    decoder.decode(tokens)
}
