#pragma once

#include "tokenizers-cpp/common.h"
#include "tokenizers-cpp/processors.rs.h"

#include <nonstd/span.hpp>
#include <string>

namespace huggingface {
namespace tokenizers {
struct Encoding {
    HFT_FFI_WRAPPER(Encoding);

public:
    bool is_empty() const noexcept { return inner_->is_empty(); }

    size_t length() const noexcept { return inner_->len(); }

    size_t size() const noexcept { return length(); }

    size_t number_of_sequences() const noexcept {
        return inner_->n_sequences();
    }

    nonstd::span<const rust::String> get_tokens() const noexcept {
        return inner_->get_tokens();
    }

    std::vector<nonstd::optional<uint32_t>> get_word_ids() const noexcept {
        auto word_ids = inner_->get_word_ids();
        std::vector<nonstd::optional<uint32_t>> result;
        fill_vec(result, word_ids, [](auto x) { return HFT_OPTION(x); });
        return result;
    }

    std::vector<nonstd::optional<size_t>> get_sequence_ids() const noexcept {
        auto sequence_ids = inner_->get_sequence_ids();
        std::vector<nonstd::optional<size_t>> result;
        fill_vec(result, sequence_ids, [](auto x) { return HFT_OPTION(x); });
        return result;
    }

    nonstd::span<const uint32_t> get_ids() const noexcept {
        return inner_->get_ids();
    }

    nonstd::span<const uint32_t> get_type_ids() const noexcept {
        return inner_->get_type_ids();
    }

    std::vector<Offsets> get_offsets() const noexcept {
        auto offsets = inner_->get_offsets();
        std::vector<Offsets> result;
        fill_vec(result, offsets);
        return result;
    }

    nonstd::span<const uint32_t> get_special_tokens_mask() const noexcept {
        return inner_->get_special_tokens_mask();
    }

    nonstd::span<const uint32_t> get_attention_mask() const noexcept {
        return inner_->get_attention_mask();
    }

    // FIXME implement correctly (may need Vec<Box<Encoding>> which isn't
    // supported in cxx yet)
    //
    // std::vector<Encoding> get_overflowing() const noexcept {
    //     auto overflowing = inner_->get_overflowing();
    //     std::vector<Encoding> result;
    //     fill_vec(result, overflowing, [](auto x) {
    //         rust::Box<ffi::Encoding> boxed(std::move(x));
    //         return Encoding(std::move(boxed));
    //     });
    //     return result;
    // }
};

struct PostProcessor {
    HFT_FFI_WRAPPER(PostProcessor);

public:
    static PostProcessor bert(nonstd::string_view sep_token, uint32_t sep_id,
                              nonstd::string_view cls_token, uint32_t cls_id) {
        return {ffi::bert_post_processor(string_view_to_str(sep_token), sep_id,
                                         string_view_to_str(cls_token),
                                         cls_id)};
    }

    static PostProcessor bert() { return bert("[SEP]", 101, "[CLS]", 102); }

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
