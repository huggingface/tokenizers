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
/**
 * @brief Represents padding parameters.
 */
struct PaddingParams {
    /**
     * @brief The padding length.
     *
     * If `nullopt`:
     * - No padding when encoding a single sequence.
     * - Pad to the longest batch element when encoding a batch.
     */
    nonstd::optional<size_t> fixed_length = nonstd::nullopt;

    /**
     * @brief Specifies the padding length.
     */
    PaddingParams& with_fixed_length(size_t length) {
        fixed_length = length;
        return *this;
    }

    /**
     * @brief Explicitly specifies no padding length is provided.
     */
    PaddingParams& with_batch_longest() {
        fixed_length = nonstd::nullopt;
        return *this;
    }

    /**
     * @brief The padding direction.
     */
    HFT_BUILDER_ARG(PaddingDirection, direction, PaddingDirection::Right);
    /**
     * @brief If specified, the padding length will always snap to the next
     * multiple of the given value.
     *
     * For example if padding length would normally be 250 but
     * `pad_to_multiple_of` is set to 8 then the actual padding length will be
     * 256.
     */
    HFT_BUILDER_ARG(size_t, pad_to_multiple_of, 0);
    /**
     * @brief The padding token id.
     */
    HFT_BUILDER_ARG(uint32_t, pad_id, 0);
    /**
     * @brief The padding type id.
     */
    HFT_BUILDER_ARG(uint32_t, pad_type_id, 0);
    /**
     * @brief The padding token string representation.
     */
    HFT_BUILDER_ARG(rust::String, pad_token, "[PAD]");
};

/**
 * @brief Represents truncation parameters.
 */
struct TruncationParams {
    /**
     * @brief The maximum length.
     */
    HFT_BUILDER_ARG(size_t, max_length, 512);
    /**
     * @brief The truncation strategy.
     *
     * - `LongestFirst`: Iteratively reduce the inputs sequence until the input
     * is under `max_length` starting from the longest one at each token (when
     * there is a pair of input sequences).
     * - `OnlyFirst`: Only truncate the first sequence.
     * - `OnlySecond`: Only truncate the second sequence.
     */
    HFT_BUILDER_ARG(TruncationStrategy, strategy,
                    TruncationStrategy::LongestFirst);
    /**
     * @brief The length of the previous first sequence to be included in the
     * overflowing sequence.
     */
    HFT_BUILDER_ARG(size_t, stride, 0);
};

/**
 * @brief A Tokenizer works as a pipeline. It converts input texts to an
 * Encoding and sequences of token ids back to text.
 *
 * The steps of the pipeline are:
 * 1. The Normalizer: in charge of normalizing the text. Common examples of
 * normalization are the unicode normalization standards, such as NFD or NFKC.
 * 2. The PreTokenizer: in charge of creating initial words splits in the text.
 * The most common way of splitting text is simply on whitespace.
 * 3. The Model: in charge of doing the actual tokenization. The only required
 * step.
 * 4. The PostProcessor: in charge of post-processing the Encoding to add
 * anything needed for the task, such as [CLS] and [SEP] tokens in a BERT
 * language model.
 */
struct Tokenizer {
    HFT_FFI_WRAPPER(Tokenizer);

private:
    // FIXME workaround for https://github.com/dtolnay/cxx/issues/496,
    //  remove together with Encoding1
    static Encoding wrap_encoding1(rust::Box<ffi::Encoding1>&& encoding_ffi) {
        return {std::move(
            reinterpret_cast<rust::Box<ffi::Encoding>&>(encoding_ffi))};
    }

    static std::vector<Encoding> wrap_encoding1_batch(
        const rust::Vec<ffi::Encoding1>& encodings_ffi) {
        std::vector<Encoding> result;
        // NOTE the line below doesn't compile under GCC/Clang with
        //  error: explicit specialization of 'size' after instantiation
        //  Not sure what makes it different from other uses of size on
        //  rust::Vec
        // result.reserve(encodings_ffi.size());
        for (const ffi::Encoding1& encoding_ffi : encodings_ffi) {
            result.push_back(wrap_encoding1(ffi::box_encoding1(encoding_ffi)));
        }
        return result;
    }

public:
    /**
     * @brief Constructs a Tokenizer from a model.
     *
     * @param model The model.
     */
    explicit Tokenizer(Model&& model) : inner_(ffi::tokenizer(*model)){};

    /**
     * @brief Specifies the normalizer.
     */
    Tokenizer& with_normalizer(Normalizer&& normalizer) {
        ffi::set_normalizer(*inner_, *normalizer);
        return *this;
    }

    /**
     * @brief Specifies the pre-tokenizer.
     */
    Tokenizer& with_pre_tokenizer(PreTokenizer&& pre_tokenizer) {
        ffi::set_pre_tokenizer(*inner_, *pre_tokenizer);
        return *this;
    }

    /**
     * @brief Specifies the post-processor.
     */
    Tokenizer& with_post_processor(PostProcessor&& post_processor) {
        ffi::set_post_processor(*inner_, *post_processor);
        return *this;
    }

    /**
     * @brief Specifies the decoder.
     */
    Tokenizer& with_decoder(Decoder&& decoder) {
        ffi::set_decoder(*inner_, *decoder);
        return *this;
    }

    /**
     * @brief Specifies the padding parameters.
     */
    Tokenizer& with_padding(PaddingParams params) {
        ffi::set_padding(*inner_, params.fixed_length.has_value(),
                         params.fixed_length.value_or(0), params.direction,
                         params.pad_to_multiple_of, params.pad_id,
                         params.pad_type_id, params.pad_token);
        return *this;
    }

    /**
     * @brief Explicitly specifies no padding (this is the default).
     */
    Tokenizer& with_no_padding() {
        ffi::set_no_padding(*inner_);
        return *this;
    }

    /**
     * @brief Specifies the truncation parameters.
     */
    Tokenizer& with_truncation(TruncationParams params) {
        ffi::set_truncation(*inner_, params.max_length, params.strategy,
                            params.stride);
        return *this;
    }

    /**
     * @brief Explicitly specifies no truncation (this is the default).
     */
    Tokenizer& with_no_truncation() {
        ffi::set_no_truncation(*inner_);
        return *this;
    }

    /**
     * @brief Returns the vocabulary size.
     *
     * @param with_added_tokens Whether to include the added tokens.
     */
    size_t get_vocab_size(bool with_added_tokens = true) {
        return ffi::get_vocab_size(*inner_, with_added_tokens);
    }

    /**
     * @brief Returns the entire vocabulary mapping (token -> ID).
     *
     * @param with_added_tokens Whether to include the added tokens.
     */
    std::unordered_map<std::string, uint32_t> get_vocab(
        bool with_added_tokens = true) {
        return ffi::vocab_to_map(ffi::get_vocab(*inner_, with_added_tokens));
    }

    /**
     * @brief Encodes a single input sequence.
     *
     * @param input The input sequence.
     * @param add_special_tokens Whether to add the special tokens
     * @param offset_type Whether byte or char offsets should be used in the
     * returned encoding.
     */
    HFT_RESULT(Encoding)
    encode(const InputSequence& input, bool add_special_tokens = true,
           OffsetType offset_type = OffsetType::Byte) {
        HFT_TRY(Encoding,
                wrap_encoding1(ffi::encode(*inner_, input, add_special_tokens,
                                           offset_type)));
    }

    /**
     * @brief Encodes a pair of input sequences.
     *
     * @param input The input sequence pair.
     * @param add_special_tokens Whether to add the special tokens
     * @param offset_type Whether byte or char offsets should be used in the
     * returned encoding.
     */
    HFT_RESULT(Encoding)
    encode_pair(const InputSequencePair& input, bool add_special_tokens,
                OffsetType offset_type = OffsetType::Byte) {
        HFT_TRY(Encoding,
                wrap_encoding1(ffi::encode_pair(
                    *inner_, input, add_special_tokens, offset_type)));
    }

    /**
     * @brief Encodes a batch of single input sequences.
     *
     * @param input The input sequences.
     * @param add_special_tokens Whether to add the special tokens
     * @param offset_type Whether byte or char offsets should be used in the
     * returned encoding.
     */
    HFT_RESULT(std::vector<Encoding>)
    encode_batch(const std::vector<InputSequence>& input,
                 bool add_special_tokens = true,
                 OffsetType offset_type = OffsetType::Byte) {
        HFT_TRY(std::vector<Encoding>,
                wrap_encoding1_batch(ffi::encode_batch(
                    *inner_, input, add_special_tokens, offset_type)));
    }

    /**
     * @brief Encodes a batch of input sequence pairs.
     *
     * @param input The input sequence pairs.
     * @param add_special_tokens Whether to add the special tokens
     * @param offset_type Whether byte or char offsets should be used in the
     * returned encoding.
     */
    HFT_RESULT(std::vector<Encoding>)
    encode_pair_batch(const std::vector<InputSequencePair>& input,
                      bool add_special_tokens = true,
                      OffsetType offset_type = OffsetType::Byte) {
        HFT_TRY(std::vector<Encoding>,
                wrap_encoding1_batch(ffi::encode_pair_batch(
                    *inner_, input, add_special_tokens, offset_type)));
    }

    /**
     * @brief Decode the given sequence of token ids back to a string.
     *
     * @param ids Token ids
     * @param skip_special_tokens Whether the special tokens should be removed
     * from the decoded string
     */
    HFT_RESULT(rust::String)
    decode(rust::Vec<uint32_t>&& ids, bool skip_special_tokens = true) {
        HFT_TRY(rust::String,
                ffi::decode(*inner_, std::move(ids), skip_special_tokens));
    }

    /**
     * @brief Decode the given sequence of token ids back to a string.
     *
     * @param ids Token ids
     * @param skip_special_tokens Whether the special tokens should be removed
     * from the decoded string
     */
    HFT_RESULT(rust::String)
    decode(nonstd::span<uint32_t> ids, bool skip_special_tokens = true) {
        rust::Vec<uint32_t> ids_vec;
        ffi::fill_vec(ids_vec, ids);
        return decode(std::move(ids_vec), skip_special_tokens);
    }

    // TODO generalize inner type of sequences
    /**
     * @brief Decode the given batch of sequences of token ids back to strings.
     *
     * @param sequences Token id sequences
     * @param skip_special_tokens Whether the special tokens should be removed
     * from the decoded strings
     */
    HFT_RESULT(rust::Vec<rust::String>)
    decode_batch(nonstd::span<std::vector<uint32_t>> sequences,
                 bool skip_special_tokens = true) {
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
