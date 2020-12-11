#pragma once

#include "tokenizers-cpp/common.h"
#include "tokenizers-cpp/normalizers.h"
#include "tokenizers-cpp/pre_tokenizers.rs.h"

#include <string>

namespace huggingface {
namespace tokenizers {
struct PreTokenizedString {
    HFT_FFI_WRAPPER(PreTokenizedString);

public:
    explicit PreTokenizedString(const NormalizedString& str)
        : inner_(ffi::normalized_to_pre_tokenized_string(*str)){};

    explicit PreTokenizedString(nonstd::string_view str)
        : inner_(ffi::str_to_pre_tokenized_string(ffi::to_rust_str(str))){};

    rust::Vec<Split> get_splits(OffsetReferential offset_ref,
                                OffsetType offset_type) {
        return ffi::get_splits(*inner_, offset_ref, offset_type);
    }
};

struct PreTokenizer {
    HFT_FFI_WRAPPER(PreTokenizer);

public:
    static PreTokenizer bert() { return {ffi::bert_pre_tokenizer()}; }

    static PreTokenizer byte_level(bool add_prefix_space, bool trim_offsets) {
        return {ffi::byte_level_pre_tokenizer(add_prefix_space, trim_offsets)};
    }

    HFT_RESULT_VOID pre_tokenize(PreTokenizedString& pre_tokenized) {
        HFT_TRY_VOID(ffi::pre_tokenize(**this, *pre_tokenized));
    }
};

}  // namespace tokenizers
}  // namespace huggingface
