#pragma once

#include "tokenizers-cpp/common.h"
#include "tokenizers-cpp/processors.rs.h"

#include <string>
// for std::pair
#include <utility>

namespace huggingface {
namespace tokenizers {
struct Encoding {
    HFT_FFI_WRAPPER(Encoding);

public:
};

struct PostProcessor {
    HFT_FFI_WRAPPER(PostProcessor);

public:
    static PostProcessor bert(std::pair<nonstd::string_view, uint32_t> sep,
                              std::pair<nonstd::string_view, uint32_t> cls) {
        return {
            ffi::bert_post_processor({to_rust_string(sep.first), sep.second},
                                     {to_rust_string(cls.first), cls.second})};
    }

    HFT_RESULT(Encoding)
    process(Encoding&& encoding, bool add_special_tokens) {
        HFT_TRY(Encoding, {ffi::process(*inner_, encoding.consume(),
                                        add_special_tokens)});
    }

    HFT_RESULT(Encoding)
    process(Encoding&& encoding, Encoding&& pair_encoding,
            bool add_special_tokens) {
        HFT_TRY(Encoding, {ffi::process_pair(*inner_, encoding.consume(),
                                             pair_encoding.consume(),
                                             add_special_tokens)});
    }
};

}  // namespace tokenizers
}  // namespace huggingface
