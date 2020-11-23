#pragma once

#include <nonstd/string_view.hpp>
#include <rust/cxx.h>

/** @file Shared code for all tokenizers-cpp modules (mostly macros) */

#define HFT_BUILDER_ARG(type, name, default_value)                    \
    type name = default_value;                                        \
    _Pragma("warning(suppress: 4458)") auto& with_##name(type name) { \
        this->name = name;                                            \
        return *this;                                                 \
    }

// Ideally we want inner_ to be private, but I couldn't make it compile
#define HFT_FFI_WRAPPER(type)                                                  \
public:                                                                        \
    type(const type&) = delete;                                                \
    type& operator=(const type&) = delete;                                     \
    type(type&&) = default;                                                    \
    type& operator=(type&&) = default;                                         \
    const ffi::type* operator->() const noexcept {                             \
        return inner_.operator->();                                            \
    }                                                                          \
    ffi::type* operator->() noexcept { return inner_.operator->(); }           \
    const ffi::type& operator*() const noexcept { return inner_.operator*(); } \
    ffi::type& operator*() noexcept { return inner_.operator*(); }             \
    rust::Box<ffi::type> inner_;                                               \
                                                                               \
private:

#if !(defined(HFT_RESULT_VOID) && defined(HFT_RESULT) && \
      defined(HFT_TRY_VOID) && defined(HFT_TRY))

#if (defined(HFT_RESULT_VOID) || defined(HFT_RESULT) || \
     defined(HFT_TRY_VOID) || defined(HFT_TRY))
#error Either all or none of HFT_RESULT_VOID, HFT_RESULT, HFT_TRY_VOID, and HFT_TRY must be defined
#endif

/// C++-side equivalent of `Result<()>`. Can be defined as e.g.
/// `std::expected<std::monostate, std::string>`. Default is `void`.
#define HFT_RESULT_VOID void

/// C++-side equivalent of `Result<T>`. Can be defined as e.g.
/// `std::expected<T, std::string>`. Default is `T`.
#define HFT_RESULT(T) T

/// Calls a Rust function returning `Result<()>`. This should invoke `expr`,
/// catch exceptions, and return a value of the type `HFT_RESULT_VOID`. E.g.
/// ```
/// try {
///     function_call;
///     return {};
/// } catch (const std::exception& e) {
///     return std::unexpected(std::string(e.what()));
/// }
/// ```
#define HFT_TRY_VOID(expr) expr;

/// Calls a Rust function returning `Result<T>`. If `HFT_RESULT`
/// is redefined, this should return `expr` on success,
/// and return a value of the corresponding type. E.g.
/// ```
/// try {
///     return function_call;
/// } catch (const std::exception& e) {
///     return std::unexpected(std::string(e.what()));
/// }
/// ```
#define HFT_TRY(T, expr) return expr;
#endif

namespace huggingface {
namespace tokenizers {
inline rust::Str string_view_to_str(nonstd::string_view string_view) {
    return {string_view.data(), string_view.size()};
}

inline nonstd::string_view str_to_string_view(rust::Str str) {
    return {str.data(), str.size()};
}
}  // namespace tokenizers
}  // namespace huggingface