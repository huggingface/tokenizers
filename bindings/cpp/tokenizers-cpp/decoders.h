#pragma once

#include "tokenizers-cpp/common.h"
#include "tokenizers-cpp/decoders.rs.h"

namespace huggingface {
namespace tokenizers {
/**
 * @brief A Decoder has the responsibility to merge the given Vec<String> into
 * a single String.
 *
 */
struct Decoder {
    HFT_FFI_WRAPPER(Decoder);

public:
    /**
     * @brief The normal BPE decoder.
     */
    static Decoder byte_level() noexcept { return {ffi::byte_level_decoder()}; }

    /**
     * @brief The WordPiece decoder takes care of decoding a list of wordpiece
     * tokens back into a readable string.
     *
     * @param prefix The prefix to be used for continuing subwords.
     * @param cleanup Whether to cleanup some tokenization artifacts (spaces
     * before punctuation, ...)
     */
    static Decoder word_piece(nonstd::string_view prefix = "##",
                              bool cleanup = true) {
        return {ffi::word_piece_decoder(ffi::to_rust_str(prefix), cleanup)};
    }

    /**
     * @brief Allows decoding Original BPE by joining all the tokens and then
     * replacing the suffix used to identify end-of-words by whitespace.
     *
     * @param suffix The suffix that was used to identify an end-of-word.
     */
    static Decoder bpe(nonstd::string_view suffix = "</w>") {
        return {ffi::bpe_decoder(ffi::to_rust_str(suffix))};
    }

    /**
     * @brief This Decoder joins the strings inserting the replacement
     * character, and then replaces it by whitespace.
     *
     * @param replacement The character replacing whitespace.
     * @param add_prefix_space Whether a leading space was added to the first
     * word during pre-tokenization (should be the same as `add_prefix_space`
     * argument of the PreTokenizer).
     */
    static Decoder metaspace(char32_t replacement = U'\u2581',
                             bool add_prefix_space = true) {
        return {ffi::metaspace_decoder(replacement, add_prefix_space)};
    }

    /**
     * @brief Decodes a sequence of strings.
     */
    HFT_RESULT(rust::String)
    decode(rust::Vec<rust::String>&& tokens) const noexcept {
        HFT_TRY(rust::String, ffi::decode_decoder(*inner_, std::move(tokens)));
    }
};

}  // namespace tokenizers
}  // namespace huggingface
