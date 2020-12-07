#pragma once

#include "tokenizers-cpp/common.h"
#include "tokenizers-cpp/normalizers.h"
#include "tokenizers-cpp/pre_tokenizers.h"
#include "tokenizers-cpp/models.h"
#include "tokenizers-cpp/processors.h"
#include "tokenizers-cpp/decoders.h"
#include "tokenizers-cpp/tokenizer.rs.h"

#include <nonstd/optional.hpp>
#include <stdexcept>

namespace huggingface {
namespace tokenizers {
struct PaddingParams {
    nonstd::optional<size_t> strategy = nonstd::nullopt;
    PaddingParams& fixed_length(size_t len) {
        strategy = len;
        return *this;
    }
    PaddingParams& batch_longest() {
        strategy = nonstd::nullopt;
        return *this;
    }

    HFT_BUILDER_ARG(PaddingDirection, direction, PaddingDirection::Left);
    HFT_BUILDER_ARG(size_t, pad_to_multiple_of, 0);
    HFT_BUILDER_ARG(uint32_t, pad_id, 0);
    HFT_BUILDER_ARG(uint32_t, pad_type_id, 0);
    HFT_BUILDER_ARG(rust::String, pad_token, "");
};

struct TruncationParams {
    HFT_BUILDER_ARG(size_t, max_length, 512);
    HFT_BUILDER_ARG(TruncationStrategy, strategy,
                    TruncationStrategy::LongestFirst);
    HFT_BUILDER_ARG(size_t, stride, 0);
};

struct Tokenizer {
    HFT_FFI_WRAPPER(Tokenizer);

public:
    explicit Tokenizer(Model&& model)
        : inner_(ffi::tokenizer(*(model.inner_))){};

    Tokenizer& with_normalizer(Normalizer&& normalizer) {
        ffi::set_normalizer(*inner_, *(normalizer.inner_));
        return *this;
    }

    Tokenizer& with_pre_tokenizer(PreTokenizer&& pre_tokenizer) {
        ffi::set_pre_tokenizer(*inner_, *(pre_tokenizer.inner_));
        return *this;
    }

    Tokenizer& with_post_processor(PostProcessor&& post_processor) {
        ffi::set_post_processor(*inner_, *(post_processor.inner_));
        return *this;
    }

    Tokenizer& with_decoder(Decoder&& decoder) {
        ffi::set_decoder(*inner_, *(decoder.inner_));
        return *this;
    }

    Tokenizer& with_padding(PaddingParams params) {
        ffi::set_padding(*inner_, params.strategy.has_value(),
                         params.strategy.value_or(0), params.direction,
                         params.pad_to_multiple_of, params.pad_id,
                         params.pad_type_id, params.pad_token);
        return *this;
    }

    Tokenizer& with_no_padding() {
        ffi::set_no_padding(*inner_);
        return *this;
    }

    Tokenizer& with_truncation(TruncationParams params) {
        ffi::set_truncation(*inner_, params.max_length, params.strategy,
                            params.stride);
        return *this;
    }

    Tokenizer& with_no_truncation() {
        ffi::set_no_truncation(*inner_);
        return *this;
    }

    HFT_RESULT(Encoding)
    encode(const InputSequence& input, bool add_special_tokens,
           OffsetType offset_type = OffsetType::Byte) {
        HFT_TRY(
            Encoding,
            // FIXME workaround for https://github.com/dtolnay/cxx/issues/496,
            //  remove together with Encoding_1
            {reinterpret_cast<rust::Box<ffi::Encoding>&&>(
                ffi::encode(*inner_, input, add_special_tokens, offset_type))});
    }

    HFT_RESULT(Encoding)
    encode_pair(const InputSequencePair& input, bool add_special_tokens,
                OffsetType offset_type = OffsetType::Byte) {
        HFT_TRY(Encoding,
                {reinterpret_cast<rust::Box<ffi::Encoding>&&>(ffi::encode_pair(
                    *inner_, input, add_special_tokens, offset_type))});
    }

    HFT_RESULT(Encoding)
    encode_batch(const std::vector<InputSequence>& input,
                 bool add_special_tokens,
                 OffsetType offset_type = OffsetType::Byte) {
        HFT_TRY(Encoding,
                {reinterpret_cast<rust::Box<ffi::Encoding>&&>(ffi::encode_batch(
                    *inner_, input, add_special_tokens, offset_type))});
    }

    HFT_RESULT(Encoding)
    encode_pair_batch(const std::vector<InputSequencePair>& input,
                      bool add_special_tokens,
                      OffsetType offset_type = OffsetType::Byte) {
        HFT_TRY(Encoding,
                {reinterpret_cast<rust::Box<ffi::Encoding>&&>(
                    ffi::encode_pair_batch(*inner_, input, add_special_tokens,
                                           offset_type))});
    }

    HFT_RESULT(rust::String)
    decode(rust::Vec<uint32_t>&& ids, bool skip_special_tokens) {
        HFT_TRY(rust::String,
                ffi::decode(*inner_, std::move(ids), skip_special_tokens));
    }

    HFT_RESULT(rust::String)
    decode(nonstd::span<uint32_t> ids, bool skip_special_tokens) {
        rust::Vec<uint32_t> ids_vec;
        fill_vec(ids_vec, ids);
        return decode(std::move(ids_vec), skip_special_tokens);
    }

    // TODO generalize inner type of sequences
    HFT_RESULT(rust::Vec<rust::String>)
    decode_batch(nonstd::span<std::vector<uint32_t>> sequences,
                 bool skip_special_tokens) {
        HFT_TRY(rust::Vec<rust::String>, [&]() {
            rust::Vec<uint32_t> sequences_vec;
            std::vector<size_t> sequence_lengths;
            for (auto& sequence : sequences) {
                for (auto id : sequence) {
                    sequences_vec.push_back(id);
                }
                sequence_lengths.push_back(sequence.size());
            }
            return ffi::decode_batch(
                *inner_, std::move(sequences_vec),
                {sequence_lengths.data(), sequence_lengths.size()},
                skip_special_tokens);
        }());
    }
};
}  // namespace tokenizers
}  // namespace huggingface
