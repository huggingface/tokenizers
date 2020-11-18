#pragma once

#include "tokenizers_util.h"
#include "normalizers.h"
#include "tokenizers-cpp/src/pre_tokenizers.rs.h"

#include <string>

namespace huggingface {
namespace tokenizers {
struct PreTokenizedString {
    FFI_WRAPPER_MEMBERS(PreTokenizedString);

public:
    explicit PreTokenizedString(NormalizedString& str)
        : inner_(ffi::pre_tokenized_string(*str)){};
};

struct BertPreTokenizer {
    FFI_WRAPPER_MEMBERS(BertPreTokenizer);

public:
    BertPreTokenizer() : inner_(ffi::bert_pre_tokenizer()){};
    void pre_tokenize(PreTokenizedString& pre_tokenized) {
        ffi::pre_tokenize_bert(*inner_, *pre_tokenized);
    }
};

}  // namespace tokenizers
}  // namespace huggingface
