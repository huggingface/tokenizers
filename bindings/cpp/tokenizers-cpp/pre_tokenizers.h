#pragma once

#include "tokenizers-cpp/common.h"
#include "tokenizers-cpp/normalizers.h"
#include "tokenizers-cpp/pre_tokenizers.rs.h"

#include <string>

namespace huggingface {
namespace tokenizers {
/**
 * @brief The PreTokenizedString is in charge of splitting an underlying string,
 making sure everything is fine while doing so, and providing ways to normalize
 and tokenize these splits.
 */
struct PreTokenizedString {
    HFT_FFI_WRAPPER(PreTokenizedString);

public:
    explicit PreTokenizedString(const NormalizedString& str) noexcept
        : inner_(ffi::normalized_to_pre_tokenized_string(*str)){};

    explicit PreTokenizedString(nonstd::string_view str)
        : inner_(ffi::str_to_pre_tokenized_string(ffi::to_rust_str(str))){};

    /**
     * @brief Returns the parts the string is split into.
     *
     * @param offset_ref Whether offsets in the original string or in the
     * normalized one are used.
     * @param offset_type Whether byte or char offsets are used.
     */
    rust::Vec<Split> get_splits(OffsetReferential offset_ref,
                                OffsetType offset_type) noexcept {
        return ffi::get_splits(*inner_, offset_ref, offset_type);
    }
};

/**
 * @brief The PreTokenizer is in charge of doing the pre-segmentation step. It
 * splits the given string in multiple substrings, keeping track of the offsets
 * of said substrings from the NormalizedString.
 */
struct PreTokenizer {
    HFT_FFI_WRAPPER(PreTokenizer);

public:
    /**
     * @brief The BERT PreTokenizer.
     */
    static PreTokenizer bert() noexcept { return {ffi::bert_pre_tokenizer()}; }

    /**
     * @brief Provides all the necessary steps to handle the BPE tokenization at
     * the byte-level. Takes care of all the required processing steps to
     * transform a UTF-8 string as needed before and after the BPE model does
     * its job.
     *
     * @param add_prefix_space Whether to add a leading space to the first word.
     * This allows to treat the leading word just as any other word.
     */
    static PreTokenizer byte_level(bool add_prefix_space = true) noexcept {
        return {ffi::byte_level_pre_tokenizer(add_prefix_space)};
    }

    /**
     * @brief This PreTokenizer splits the string by the given character.
     *
     * @param delimiter The character to split on.
     */
    static HFT_RESULT(PreTokenizer) char_delimiter(char32_t delimiter) {
        HFT_TRY(PreTokenizer, {ffi::char_delimiter_pre_tokenizer(delimiter)});
    }

    /**
     * @brief This PreTokenizer replaces all the whitespaces by the provided
     * meta character and then splits on this character.
     *
     * @param replacement The character replacing whitespace.
     * @param add_prefix_space Whether to add a leading space to the first word.
     * This allows to treat the leading word just as any other word.
     */
    static HFT_RESULT(PreTokenizer) metaspace(char32_t replacement = U'\u2581',
                                              bool add_prefix_space = true) {
        HFT_TRY(PreTokenizer,
                {ffi::metaspace_pre_tokenizer(replacement, add_prefix_space)});
    }

    /**
     * @brief This PreTokenizer splits the string into parts matching
     * \w+|[^\w\s]+ and removes all whitespace.
     *
     * Compared to `whitespace_split`, separates `What?` into `What` and `?`.
     */
    static PreTokenizer whitespace() {
        return {ffi::whitespace_pre_tokenizer()};
    }

    /**
     * @brief This PreTokenizer splits on a literal string.
     *
     * @param pattern The string to split on.
     * @param behavior Defines the expected behavior for the delimiter.
     * For example, when splitting on `'-'` with input `the-final--countdown`:
     * - Removed => `[ "the", "final", "countdown" ]`
     * - Isolated => `[ "the", "-", "final", "-", "-", "countdown" ]`
     * - MergedWithPrevious => `[ "the-", "final-", "-", "countdown" ]`
     * - MergedWithNext => `[ "the", "-final", "-", "-countdown" ]`
     * - Contiguous => `[ "the", "-", "final", "--", "countdown" ]`
     * @param invert Whether matches should be inverted (i.e. the pattern
     * matches words, not delimiters).
     */
    static PreTokenizer split_literal(nonstd::string_view pattern,
                                      SplitDelimiterBehavior behavior,
                                      bool invert = false) {
        return {ffi::split_literal_pre_tokenizer(ffi::to_rust_str(pattern),
                                                 behavior, invert)};
    }

    /**
     * @brief This PreTokenizer splits on a regular expression.
     *
     * @param pattern The pattern to split on (uses [Rust regex
     * syntax](https://docs.rs/regex/1.4.2/regex/#syntax), not C++!)
     * @param behavior Defines the expected behavior for the delimiter.
     * For example, when splitting on `'-'` with input `the-final--countdown`:
     * - Removed => `[ "the", "final", "countdown" ]`
     * - Isolated => `[ "the", "-", "final", "-", "-", "countdown" ]`
     * - MergedWithPrevious => `[ "the-", "final-", "-", "countdown" ]`
     * - MergedWithNext => `[ "the", "-final", "-", "-countdown" ]`
     * - Contiguous => `[ "the", "-", "final", "--", "countdown" ]`
     * @param invert Whether matches should be inverted (i.e. the pattern
     * matches words, not delimiters).
     */
    static HFT_RESULT(PreTokenizer)
        split_regex(nonstd::string_view pattern,
                    SplitDelimiterBehavior behavior, bool invert) {
        HFT_TRY(PreTokenizer,
                {ffi::split_regex_pre_tokenizer(ffi::to_rust_str(pattern),
                                                behavior, invert)});
    }

    /**
     * @brief This PreTokenizer splits on code points belonging to a
     * "Punctuation" [Unicode category](http://www.unicode.org/notes/tn36/): Pc,
     * Pd, Pe, Pf, Pi, Po, or Ps.
     */
    static PreTokenizer punctuation() {
        return {ffi::punctuation_pre_tokenizer()};
    }

    /**
     * @brief This PreTokenizer splits on whitespace.
     *
     * Compared to `whitespace`, `What?` is kept as a single part.
     */
    static PreTokenizer whitespace_split() {
        return {ffi::whitespace_split_pre_tokenizer()};
    }

    /**
     * @brief This PreTokenizer splits numbers into single tokens.
     *
     * @param individual_digits If true, all digits are split into individual
     * tokens.
     */
    static PreTokenizer digits(bool individual_digits = false) {
        return {ffi::digits_pre_tokenizer(individual_digits)};
    }

    /**
     * @brief This PreTokenizer splits into parts belonging to different
     * [Unicode scripts](http://www.unicode.org/reports/tr24/tr24-31.html).
     *
     * For consistency with Google SentencePiece implementation, Hiragana,
     * Katakana, and Han are treated as a single script.
     *
     * @return PreTokenizer
     */
    static PreTokenizer unicode_scripts() {
        return {ffi::unicode_scripts_pre_tokenizer()};
    }

    /**
     * @brief This PreTokenizer represents a sequence of other pre-tokenizers
     *
     * Use this overload when the sequence is built dynamically, e.g.
     *
     * ```c++
     * std::vector<PreTokenizer> pre_tokenizers;
     * pre_tokenizers.push_back(a_pre_tokenizer);
     * if (some_condition) {
     *     pre_tokenizers.push_back(another_pre_tokenizer);
     * }
     * PreTokenizer::sequence(pre_tokenizers);
     * ```
     *
     * @param pre_tokenizers The pre_tokenizers
     */
    static PreTokenizer sequence(std::vector<PreTokenizer>& pre_tokenizers) {
        rust::Box<ffi::PreTokenizerVec> pre_tokenizers_ffi{
            ffi::init_pre_tokenizer_vec()};
        for (PreTokenizer& pre_tokenizer : pre_tokenizers) {
            ffi::add_pre_tokenizer(*pre_tokenizers_ffi,
                                   pre_tokenizer.consume());
        }
        return {ffi::sequence_pre_tokenizer(std::move(pre_tokenizers_ffi))};
    }

    /**
     * @brief This PreTokenizer represents a sequence of other pre_tokenizers
     *
     * Prefer this overload the sequence is known in advance, e.g.
     * `PreTokenizer::sequence(PreTokenizer::nfc(), PreTokenizer::lowercase())`.
     *
     * @param pre_tokenizers The pre_tokenizers (all of type PreTokenizer&&)
     */
    template <typename... Args>
    static PreTokenizer sequence(Args&&... pre_tokenizers) {
        std::vector<PreTokenizer> holder;

        ffi::vararg_for([&](auto&& arg) { holder.push_back(std::move(arg)); },
                        std::forward<Args>(pre_tokenizers)...);

        return sequence(holder);
    }

    /**
     * @brief Applies this PreTokenizer to the argument.
     *
     * @param pre_tokenized The PreTokenizedString to be pretokenized.
     */
    HFT_RESULT_VOID pre_tokenize(PreTokenizedString& pre_tokenized) const {
        HFT_TRY_VOID(ffi::pre_tokenize(**this, *pre_tokenized));
    }
};

}  // namespace tokenizers
}  // namespace huggingface
