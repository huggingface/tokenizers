#pragma once

#include "tokenizers-cpp/common.h"
#include "tokenizers-cpp/normalizers.rs.h"

#include <string>

namespace huggingface {
namespace tokenizers {
struct NormalizedString {
    HFT_FFI_WRAPPER(NormalizedString);

public:
    explicit NormalizedString(nonstd::string_view str)
        : inner_(ffi::normalized_string(to_rust_str(str))) {}

    operator nonstd::string_view() { return get_normalized(); }

    nonstd::string_view get_normalized() {
        return to_string_view(ffi::get_normalized(*inner_));
    }

    nonstd::string_view get_original() {
        return to_string_view(ffi::get_original(*inner_));
    }
};

struct Normalizer {
    HFT_FFI_WRAPPER(Normalizer);

public:
    static Normalizer bert(bool clean_text, bool handle_chinese_chars,
                           BertStripAccents strip_accents, bool lowercase) {
        return {ffi::bert_normalizer(clean_text, handle_chinese_chars,
                                     strip_accents, lowercase)};
    }

    HFT_RESULT_VOID normalize(NormalizedString& normalized) {
        HFT_TRY_VOID(ffi::normalize(*inner_, *normalized));
    }
};

struct BertNormalizerOptions {
    HFT_BUILDER_ARG(bool, clean_text, true);
    HFT_BUILDER_ARG(bool, handle_chinese_chars, true);
    HFT_BUILDER_ARG(bool, lowercase, true);

    BertStripAccents strip_accents = BertStripAccents::DeterminedByLowercase;
    HFT_DISABLE_WARNING_PUSH
    HFT_DISABLE_WARNING(-Wshadow, 4458)
    BertNormalizerOptions& with_strip_accents(bool strip_accents) {
        this->strip_accents =
            strip_accents ? BertStripAccents::True : BertStripAccents::False;
        return *this;
    }
    HFT_DISABLE_WARNING_POP

    Normalizer build() {
        return Normalizer::bert(clean_text, handle_chinese_chars, strip_accents,
                                lowercase);
    }
};

}  // namespace tokenizers
}  // namespace huggingface
