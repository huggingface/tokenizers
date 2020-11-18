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

struct BertNormalizerOptions {
    BUILDER_ARG(bool, clean_text, true);
    BUILDER_ARG(bool, handle_chinese_chars, true);
    BUILDER_ARG(bool, lowercase, true);

    BertStripAccents strip_accents = BertStripAccents::DeterminedByLowercase;
    BertNormalizerOptions& with_strip_accents(bool strip_accents) {
        this->strip_accents =
            strip_accents ? BertStripAccents::True : BertStripAccents::False;
        return *this;
    }

    BertNormalizer build() {
        return {ffi::bert_normalizer(clean_text, handle_chinese_chars,
                                     strip_accents, lowercase)};
    }
};

}  // namespace tokenizers
}  // namespace huggingface
