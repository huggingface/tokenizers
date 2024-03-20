#pragma once

#include "tokenizers-cpp/common.h"
#include "tokenizers-cpp/normalizers.rs.h"

#include <nonstd/span.hpp>
#include <string>

namespace huggingface {
namespace tokenizers {
/**
 * @brief A NormalizedString takes care of processing an "original" string to
 * modify it and obtain a "normalized" string.
 */
struct NormalizedString {
    HFT_FFI_WRAPPER(NormalizedString);

public:
    /**
     * @brief Construct a new NormalizedString object.
     *
     * @param str The original string
     */
    explicit NormalizedString(nonstd::string_view str) noexcept
        : inner_(ffi::normalized_string(ffi::to_rust_str(str))) {}

    /**
     * @brief Returns the normalized string.
     */
    operator nonstd::string_view() noexcept { return get_normalized(); }

    /**
     * @brief Returns the normalized string.
     */
    nonstd::string_view get_normalized() noexcept {
        return ffi::to_string_view(ffi::get_normalized(*inner_));
    }

    /**
     * @brief Returns the original string.
     */
    nonstd::string_view get_original() noexcept {
        return ffi::to_string_view(ffi::get_original(*inner_));
    }
};

/**
 * @brief Options for BERT normalizer (see Normalizer::bert()).
 */
struct BertOptions {
    HFT_BUILDER_ARG(bool, clean_text, true);
    HFT_BUILDER_ARG(bool, handle_chinese_chars, true);
    HFT_BUILDER_ARG(bool, lowercase, true);

    BertStripAccents strip_accents = BertStripAccents::IfNotLowercase;
    HFT_DISABLE_WARNING_PUSH
    HFT_DISABLE_WARNING(-Wshadow, 4458)
    /**
     * @brief Sets whether the accents should be stripped. By default they are
     * stripped if `lowercase` is `true`, not stripped otherwise.
     */
    BertOptions& with_strip_accents(bool strip_accents) {
        this->strip_accents =
            strip_accents ? BertStripAccents::True : BertStripAccents::False;
        return *this;
    }
    HFT_DISABLE_WARNING_POP
};

/**
 * @brief Takes care of string pre-processing.
 */
struct Normalizer {
    HFT_FFI_WRAPPER(Normalizer);

public:
    /**
     * @brief BERT Normalizer.
     *
     * @param clean_text Whether to do the bert basic cleaning:
     *   1. Remove any control characters
     *   2. Replace all sorts of whitespace by the classic one ` `
     * @param handle_chinese_chars Whether to split each Chinese character into
     * a separate token
     * @param strip_accents Whether to strip accents
     * @param lowercase Whether to lowercase the input
     */
    static Normalizer bert(
        bool clean_text = true, bool handle_chinese_chars = true,
        BertStripAccents strip_accents = BertStripAccents::IfNotLowercase,
        bool lowercase = true) noexcept {
        return {ffi::bert_normalizer(clean_text, handle_chinese_chars,
                                     strip_accents, lowercase)};
    }

    /**
     * @brief BERT Normalizer (overload which takes the options struct).
     *
     * @param options BERT options
     */
    static Normalizer bert(BertOptions options) noexcept {
        return bert(options.clean_text, options.handle_chinese_chars,
                    options.strip_accents, options.lowercase);
    }

    /**
     * @brief This Normalizer strips whitespace from string ends.
     *
     * @param strip_left Whether to strip whitespace on the left
     * @param strip_right Whether to strip whitespace on the right
     */
    static Normalizer strip(bool strip_left = true,
                            bool strip_right = true) noexcept {
        return {ffi::strip_normalizer(strip_left, strip_right)};
    }

    /**
     * @brief This Normalizer removes combining marks.
     */
    static Normalizer strip_accents() noexcept {
        return {ffi::strip_accents_normalizer()};
    }

    /**
     * @brief This Normalizer applies
     * [https://unicode.org/reports/tr15/#Norm_Forms](NFC Unicode normalization
     * form).
     */
    static Normalizer nfc() noexcept { return {ffi::nfc_normalizer()}; }

    /**
     * @brief This Normalizer applies
     * [https://unicode.org/reports/tr15/#Norm_Forms](NFD Unicode normalization
     * form).
     */
    static Normalizer nfd() noexcept { return {ffi::nfd_normalizer()}; }

    /**
     * @brief This Normalizer applies
     * [https://unicode.org/reports/tr15/#Norm_Forms](NFKC Unicode normalization
     * form).
     */
    static Normalizer nfkc() noexcept { return {ffi::nfkc_normalizer()}; }

    /**
     * @brief This Normalizer applies
     * [https://unicode.org/reports/tr15/#Norm_Forms](NFKD Unicode normalization
     * form).
     */
    static Normalizer nfkd() noexcept { return {ffi::nfkd_normalizer()}; }

    /**
     * @brief This Normalizer lowercases the string
     */
    static Normalizer lowercase() noexcept {
        return {ffi::lowercase_normalizer()};
    }

    /**
     * @brief This Normalizer replaces all occurences of a string
     *
     * @param pattern The string to be replaced
     * @param replacement The replacement
     */
    static Normalizer replace_literal(nonstd::string_view pattern,
                                      nonstd::string_view replacement) {
        return {ffi::replace_literal_normalizer(ffi::to_rust_str(pattern),
                                                ffi::to_rust_str(replacement))};
    }

    /**
     * @brief This Normalizer replaces all matches of a regular expression
     *
     * @param pattern The pattern to be replaced (uses [Rust regex
     * syntax](https://docs.rs/regex/1.4.2/regex/#syntax), not C++!)
     * @param replacement The replacement
     */
    static HFT_RESULT(Normalizer)
        replace_regex(nonstd::string_view pattern,
                      nonstd::string_view replacement) {
        HFT_TRY(Normalizer,
                {ffi::replace_regex_normalizer(ffi::to_rust_str(pattern),
                                               ffi::to_rust_str(replacement))});
    }

    /**
     * @brief This Normalizer represents a sequence of other normalizers
     *
     * Use this overload when the sequence is built dynamically, e.g.
     *
     * ```c++
     * std::vector<Normalizer> normalizers;
     * normalizers.push_back(a_normalizer);
     * if (some_condition) {
     *     normalizers.push_back(another_normalizer);
     * }
     * Normalizer::sequence(normalizers);
     * ```
     *
     * @param normalizers The normalizers
     */
    static Normalizer sequence(std::vector<Normalizer>& normalizers) {
        rust::Box<ffi::NormalizerVec> normalizers_ffi{
            ffi::init_normalizer_vec()};
        for (Normalizer& normalizer : normalizers) {
            ffi::add_normalizer(*normalizers_ffi, normalizer.consume());
        }
        return {ffi::sequence_normalizer(std::move(normalizers_ffi))};
    }

    /**
     * @brief This Normalizer represents a sequence of other normalizers
     *
     * Prefer this overload the sequence is known in advance, e.g.
     * `Normalizer::sequence(Normalizer::nfc(), Normalizer::lowercase())`.
     *
     * @param normalizers The normalizers (all of type Normalizer&&)
     */
    template <typename... Args>
    static Normalizer sequence(Args&&... normalizers) {
        std::vector<Normalizer> holder;

        ffi::vararg_for([&](auto&& arg) { holder.push_back(std::move(arg)); },
                        std::forward<Args>(normalizers)...);

        return sequence(holder);
    }

    /**
     * @brief Applies this normalizer to the argument.
     *
     * @param normalized The NormalizedString to be normalized
     */
    HFT_RESULT_VOID
    normalize(NormalizedString& normalized) const {
        HFT_TRY_VOID(ffi::normalize(*inner_, *normalized));
    }
};
}  // namespace tokenizers
}  // namespace huggingface
