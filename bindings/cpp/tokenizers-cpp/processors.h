#pragma once

#include "tokenizers-cpp/common.h"
#include "tokenizers-cpp/processors.rs.h"

#include <nonstd/span.hpp>
#include <string>

namespace huggingface {
namespace tokenizers {
/**
 * @brief Represents the output of a Tokenizer.
 */
struct Encoding {
    HFT_FFI_WRAPPER(Encoding);

public:
    /**
     * @brief Returns whether this Encoding is empty.
     */
    bool is_empty() const noexcept { return inner_->is_empty(); }

    /**
     * @brief Returns the total length of this Encoding.
     */
    size_t length() const noexcept { return inner_->len(); }

    /**
     * @brief Returns the total length of this Encoding (same as
     * Encoding::length()).
     */
    size_t size() const noexcept { return length(); }

    /**
     * @brief Returns the number of sequences combined in this Encoding.
     */
    size_t number_of_sequences() const noexcept {
        return inner_->n_sequences();
    }

    /**
     * @brief Returns tokens in this Encoding.
     */
    nonstd::span<const rust::String> get_tokens() const noexcept {
        return inner_->get_tokens();
    }

    /**
     * @brief Returns word ids in this Encoding.
     */
    std::vector<nonstd::optional<uint32_t>> get_word_ids() const noexcept {
        auto word_ids = inner_->get_word_ids();
        std::vector<nonstd::optional<uint32_t>> result;
        ffi::fill_vec(result, word_ids,
                      [](auto x) { return ffi::to_optional(x); });
        return result;
    }

    /**
     * @brief Returns sequence ids in this Encoding.
     */
    std::vector<nonstd::optional<size_t>> get_sequence_ids() const noexcept {
        auto sequence_ids = inner_->get_sequence_ids();
        std::vector<nonstd::optional<size_t>> result;
        ffi::fill_vec(result, sequence_ids,
                      [](auto x) { return ffi::to_optional(x); });
        return result;
    }

    /**
     * @brief Returns token ids in this Encoding.
     */
    nonstd::span<const uint32_t> get_ids() const noexcept {
        return inner_->get_ids();
    }

    /**
     * @brief Returns type ids in this Encoding.
     */
    nonstd::span<const uint32_t> get_type_ids() const noexcept {
        return inner_->get_type_ids();
    }

    /**
     * @brief Returns offsets in this Encoding.
     */
    std::vector<Offsets> get_offsets() const noexcept {
        auto offsets = inner_->get_offsets();
        std::vector<Offsets> result;
        ffi::fill_vec(result, offsets);
        return result;
    }

    /**
     * @brief Returns the special tokens mask in this Encoding.
     */
    nonstd::span<const uint32_t> get_special_tokens_mask() const noexcept {
        return inner_->get_special_tokens_mask();
    }

    /**
     * @brief Returns the attention mask in this Encoding.
     */
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

/**
 * @brief A PostProcessor has the responsibility to post-process an encoded
 * output of the Tokenizer. It adds any required special tokens.
 *
 */
struct PostProcessor {
    HFT_FFI_WRAPPER(PostProcessor);

public:
    /**
     * @brief A BERT PostProcessor.
     *
     * @param sep_token Token between and at the end of sequences.
     * @param sep_id Token id between and at the end of sequences.
     * @param cls_token Token for the beginning of the first sequence.
     * @param cls_id Token id for the beginning of the first sequence.
     */
    static PostProcessor bert(nonstd::string_view sep_token, uint32_t sep_id,
                              nonstd::string_view cls_token, uint32_t cls_id) {
        return {ffi::bert_post_processor(ffi::to_rust_str(sep_token), sep_id,
                                         ffi::to_rust_str(cls_token), cls_id)};
    }

    /**
     * @brief The default BERT PostProcessor.
     */
    static PostProcessor bert() { return bert("[SEP]", 101, "[CLS]", 102); }

    /**
     * @brief A BPE PostProcessor.
     *
     * @param add_prefix_space Whether a leading space was added to the first
     * word during pre-tokenization (should be the same as `add_prefix_space`
     * argument of the PreTokenizer).
     * @param trim_offsets Whether to trim offsets to avoid including
     * whitespaces.
     */
    static PostProcessor byte_level(bool add_prefix_space = true,
                                    bool trim_offsets = true) {
        return {ffi::byte_level_post_processor(add_prefix_space, trim_offsets)};
    }

    /**
     * @brief A RoBERTa PostProcessor.
     *
     * @param sep_token Token between and at the end of sequences.
     * @param sep_id Token id between and at the end of sequences.
     * @param cls_token Token for the beginning of the first sequence.
     * @param cls_id Token id for the beginning of the first sequence.
     * @param add_prefix_space Whether a leading space was added to the first
     * word during pre-tokenization (should be the same as `add_prefix_space`
     * argument of the PreTokenizer).
     * @param trim_offsets Whether to trim offsets to avoid including
     * whitespaces.
     */
    static PostProcessor roberta(nonstd::string_view sep_token, uint32_t sep_id,
                                 nonstd::string_view cls_token, uint32_t cls_id,
                                 bool add_prefix_space = true,
                                 bool trim_offsets = true) {
        return {ffi::roberta_post_processor(ffi::to_rust_str(sep_token), sep_id,
                                            ffi::to_rust_str(cls_token), cls_id,
                                            add_prefix_space, trim_offsets)};
    }

    /**
     * @brief The default RoBERTa PostProcessor.
     */
    static PostProcessor roberta() {
        return roberta("</s>", 2, "<s>", 0, true, true);
    }

    /**
     * @brief Process an encoding.
     */
    HFT_RESULT(Encoding)
    process(Encoding&& encoding, bool add_special_tokens = true) {
        HFT_TRY(Encoding, {ffi::process(*inner_, encoding.consume(),
                                        add_special_tokens)});
    }

    /**
     * @brief Process and merge two encodings.
     */
    HFT_RESULT(Encoding)
    process(Encoding&& encoding, Encoding&& pair_encoding,
            bool add_special_tokens = true) {
        HFT_TRY(Encoding, {ffi::process_pair(*inner_, encoding.consume(),
                                             pair_encoding.consume(),
                                             add_special_tokens)});
    }
};

/**
 * @brief Configuration class for template processing.
 *
 * ## Example
 *
 * Let's take BERT tokenizer as an example. It uses two special tokens, used to
 * delimitate each sequence. `[CLS]` is always used at the beginning of the
first
 * sequence, and `[SEP]` is added at the end of both the first, and the pair
 * sequences. The final result looks like this:
 * - Single sequence: `[CLS] Hello there [SEP]`
 * - Pair sequences: `[CLS] My name is Anthony [SEP] What is my name? [SEP]`
 * With the type ids as following:
 * ```markdown
 * [CLS]   ...   [SEP]   ...   [SEP]
 *   0      0      0      1      1
 * ```
 *
 * So, we can define a TemplateProcessingBuilder that will achieve this result:
 * ```c++
 * TemplateProcessingBuilder()
 *     .single("[CLS] $0 [SEP]")
 *     .pair("[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1")
 *     .special_tokens({{"[CLS]", 101}, {"[SEP]", 102}})
 *     .build();
 * ```
 *
 * In this example, each input sequence is identified using a `$` construct.
 * This identifier lets us specify each input sequence, and the type_id to use.
 * When nothing is specified, it uses the default values. Here are the different
 * ways to specify it:
 * - Specifying the sequence, with default `type_id == 0`: `$A` or `$B`
 * - Specifying the `type_id` with default `sequence == A`: `$0`, `$1`, `$2`,
 *  ...
 * - Specifying both: `$A:0`, `$B:1`, ...
 *
 * The same construct is used for special tokens: `<identifier>(:<type_id>)?`.
 *
 * **Warning**: You must ensure that you are giving the correct tokens/ids as
 * these will be added to the Encoding without any further check. If the given
 * ids correspond to something totally different in a Tokenizer using this
 * PostProcessor, it might lead to unexpected results.
 */
class TemplateProcessingBuilder {
    HFT_FFI_WRAPPER(TemplateProcessingBuilder);

public:
    /**
     * @brief The default constructor.
     */
    TemplateProcessingBuilder() noexcept
        : inner_(ffi::template_processing_builder()){};

    /**
     * @brief Specifies the template for a single sequence. See
     * class documentation for details and an example.
     */
    HFT_RESULT(HFT_REF(TemplateProcessingBuilder))
    single(nonstd::string_view sequence_template) {
        HFT_TRY(HFT_REF(TemplateProcessingBuilder),
                [&]() -> HFT_REF(TemplateProcessingBuilder) {
                    inner_->single(ffi::to_rust_str(sequence_template));
                    return HFT_WRAP_REF(*this);
                }());
    }

    /**
     * @brief Specifies the template for a pair of sequences. See
     * class documentation for details and an example.
     */
    HFT_RESULT(HFT_REF(TemplateProcessingBuilder))
    pair(nonstd::string_view sequence_template) {
        HFT_TRY(HFT_REF(TemplateProcessingBuilder),
                [&]() -> HFT_REF(TemplateProcessingBuilder) {
                    inner_->pair(ffi::to_rust_str(sequence_template));
                    return HFT_WRAP_REF(*this);
                }());
    }

    /**
     * @brief Specifies special tokens. See class documentation for details and
     * an example.
     */
    TemplateProcessingBuilder& special_tokens(
        nonstd::span<const SpecialToken> tokens) noexcept {
        inner_->special_tokens(ffi::to_rust_slice(tokens));
        return *this;
    }

    /**
     * @brief Specifies special tokens. See class documentation for details and
     * an example.
     */
    TemplateProcessingBuilder& special_tokens(
        std::initializer_list<SpecialToken> tokens) noexcept {
        return special_tokens({tokens.begin(), tokens.size()});
    }

    /**
     * @brief Builds the PostProcessor for the given template(s).
     */
    HFT_RESULT(PostProcessor) build() {
        HFT_TRY(PostProcessor, PostProcessor(inner_->build()));
    }
};

}  // namespace tokenizers
}  // namespace huggingface
