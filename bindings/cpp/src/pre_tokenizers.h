#pragma once

#include "tokenizers_util.h"
#include "normalizers.h"
#include "tokenizers-cpp/src/pre_tokenizers.rs.h"

#include <string>

namespace huggingface {
namespace tokenizers {
struct PreTokenizedString {
    rust::Box<ffi::PreTokenizedString> inner_;
    DELETE_COPY(PreTokenizedString);

    static PreTokenizedString from(NormalizedString& str) {
        return {ffi::pre_tokenized_string(*str.inner_)};
    }
};

struct BertPreTokenizer {
    rust::Box<ffi::BertPreTokenizer> inner_;
    DELETE_COPY(BertPreTokenizer);

    void pre_tokenize(PreTokenizedString& pre_tokenized) {
        ffi::pre_tokenize_bert(*inner_, *pre_tokenized.inner_);
    }
};

}  // namespace tokenizers
}  // namespace huggingface
