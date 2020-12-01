#[cxx::bridge(namespace = "huggingface::tokenizers")]
mod ffi {
    extern "C++" {
        include!("tokenizers-cpp/decoders.h");
    }

    #[namespace = "huggingface::tokenizers::ffi"]
    extern "Rust" {
        type DecoderWrapper;
        type ByteLevelDecoder;

        fn byte_level_decoder(add_prefix_space: bool, trim_offsets: bool) -> Box<ByteLevelDecoder>;

        fn decode_byte_level(decoder: &ByteLevelDecoder, tokens: Vec<String>) -> Result<String>;
    }
}

use derive_more::{Deref, DerefMut, From, Into};
use tk::{decoders::byte_level::ByteLevel as TkByteLevelDecoder, Decoder, Result};

#[derive(Deref, DerefMut, From, Into)]
struct DecoderWrapper(tk::DecoderWrapper);
#[derive(Deref, DerefMut, From, Into)]
struct ByteLevelDecoder(TkByteLevelDecoder);

fn byte_level_decoder(add_prefix_space: bool, trim_offsets: bool) -> Box<ByteLevelDecoder> {
    Box::new(ByteLevelDecoder(TkByteLevelDecoder::new(
        add_prefix_space,
        trim_offsets,
    )))
}

fn decode_byte_level(decoder: &ByteLevelDecoder, tokens: Vec<String>) -> Result<String> {
    decoder.decode(tokens)
}
