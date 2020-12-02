#pragma once

#include "tokenizers-cpp/common.h"
#include "tokenizers-cpp/decoders.rs.h"

namespace huggingface {
namespace tokenizers {
struct ByteLevelDecoder {
    HFT_FFI_WRAPPER(ByteLevelDecoder);

public:
    ByteLevelDecoder(bool add_prefix_space, bool trim_offsets)
        : inner_(ffi::byte_level_decoder(add_prefix_space, trim_offsets)){};

    HFT_RESULT(rust::String) decode(rust::Vec<rust::String>&& tokens) {
        HFT_TRY(rust::String,
                ffi::decode_byte_level(*inner_, std::move(tokens)));
    }
};

struct DecoderWrapper {
    HFT_FFI_WRAPPER(DecoderWrapper);

public:
};

}  // namespace tokenizers
}  // namespace huggingface
