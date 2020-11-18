#pragma once

#include "tokenizers_util.h"
#include "normalizers.h"
#include "tokenizers-cpp/src/pre_tokenizers.rs.h"

#include <string>

namespace huggingface {
namespace tokenizers {
struct PreTokenizedString {
    HFT_FFI_WRAPPER(PreTokenizedString);

public:
    static HFT_RESULT(PreTokenizedString) from(const NormalizedString& str) {
        HFT_TRY(PreTokenizedString, {ffi::pre_tokenized_string(*str)});
    }
};

struct BertPreTokenizer {
    HFT_FFI_WRAPPER(BertPreTokenizer);

public:
    BertPreTokenizer() : inner_(ffi::bert_pre_tokenizer()){};

    HFT_RESULT_VOID pre_tokenize(PreTokenizedString& pre_tokenized) {
        HFT_TRY_VOID(ffi::pre_tokenize_bert(*inner_, *pre_tokenized));
    }
};

}  // namespace tokenizers
}  // namespace huggingface
