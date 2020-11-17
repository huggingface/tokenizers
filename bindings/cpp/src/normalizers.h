#pragma once

#include "tokenizers_util.h"
#include "tokenizers-cpp/src/normalizers.rs.h"

#include <string>

namespace huggingface {
namespace tokenizers {
struct NormalizedString {
    rust::Box<ffi::NormalizedString> inner_;
    DELETE_COPY(NormalizedString);

    static NormalizedString from(const std::string& str) {
        return {ffi::normalized_string(str)};
    }
};

struct BertNormalizer {
    rust::Box<ffi::BertNormalizer> inner_;
    DELETE_COPY(BertNormalizer);

    void normalize(NormalizedString& normalized) {
        ffi::normalize_bert(*inner_, *normalized.inner_);
    }
};

class BertNormalizerOptions {
    BUILDER_ARG(bool, clean_text, true);
    BUILDER_ARG(bool, handle_chinese_chars, true);
    BUILDER_ARG(bool, lowercase, true);

private:
    BertStripAccents strip_accents_ = BertStripAccents::DeterminedByLowercase;

public:
    BertNormalizerOptions& strip_accents(bool strip_accents) {
        this->strip_accents_ =
            strip_accents ? BertStripAccents::True : BertStripAccents::False;
        return *this;
    }

    BertNormalizer build() {
        return {ffi::bert_normalizer(clean_text_, handle_chinese_chars_,
                                     strip_accents_, lowercase_)};
    }
};

}  // namespace tokenizers
}  // namespace huggingface
