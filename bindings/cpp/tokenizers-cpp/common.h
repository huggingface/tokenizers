#pragma once

#include <nonstd/optional.hpp>
#include <nonstd/string_view.hpp>
#include <nonstd/span.hpp>
#include <rust/cxx.h>
#include <initializer_list>

/** @file common.h Shared code for all tokenizers-cpp modules (mostly macros) */

// code adapted from
// https://www.fluentcpp.com/2019/08/30/how-to-disable-a-warning-in-cpp/
// to disable warnings in a cross-compiler way
#if defined(_MSC_VER)
#define HFT_DISABLE_WARNING_PUSH __pragma(warning(push))
#define HFT_DISABLE_WARNING_POP __pragma(warning(pop))
#define HFT_DISABLE_WARNING(gcc_name, msvc_numbers) \
    __pragma(warning(disable : msvc_numbers))

#elif defined(__GNUC__)
#define HFT_DO_PRAGMA(X) _Pragma(#X)
#define HFT_DISABLE_WARNING_PUSH HFT_DO_PRAGMA(GCC diagnostic push)
#define HFT_DISABLE_WARNING_POP HFT_DO_PRAGMA(GCC diagnostic pop)
#define HFT_DISABLE_WARNING(gcc_name, msvc_numbers) \
    HFT_DO_PRAGMA(GCC diagnostic ignored #gcc_name)

#else
#define HFT_DISABLE_WARNING_PUSH
#define HFT_DISABLE_WARNING_POP
#define HFT_DISABLE_WARNING(gcc_name, msvc_numbers)

#endif

/**
 * @def HFT_DISABLE_WARNING_PUSH
 *
 * @brief Begins a section where some warnings are disabled
 */

/**
 * @def HFT_DISABLE_WARNING_POP
 *
 * @brief Ends a section where some warnings are disabled
 */

/**
 * @def HFT_DISABLE_WARNING
 *
 * @brief Disables specific warnings
 */

/**
 * @brief For builder members
 */
#define HFT_BUILDER_ARG(type, name, default_value) \
    type name = default_value;                     \
    HFT_DISABLE_WARNING_PUSH                       \
    HFT_DISABLE_WARNING(-Wshadow, 4458)            \
    auto& with_##name(type name) {                 \
        this->name = name;                         \
        return *this;                              \
    }                                              \
    HFT_DISABLE_WARNING_POP

/**
 * @brief Declares standard members of every FFI wrapper
 *
 * Disables copy constructor/assignment, enables move constructor/assignment,
 * and dereference operators.
 */
#define HFT_FFI_WRAPPER(type)                                                  \
public:                                                                        \
    type(rust::Box<ffi::type>&& inner) : inner_(std::move(inner)) {};          \
    type(const type&) = delete;                                                \
    type& operator=(const type&) = delete;                                     \
    type(type&&) noexcept = default;                                           \
    type& operator=(type&&) noexcept = default;                                \
    const ffi::type* operator->() const noexcept {                             \
        return inner_.operator->();                                            \
    }                                                                          \
    ffi::type* operator->() noexcept { return inner_.operator->(); }           \
    const ffi::type& operator*() const noexcept { return inner_.operator*(); } \
    ffi::type& operator*() noexcept { return inner_.operator*(); }             \
    rust::Box<ffi::type>&& consume() { return std::move(inner_); }             \
                                                                               \
private:                                                                       \
    rust::Box<ffi::type> inner_;

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
namespace ffi {
/**
 * @brief Converts an FFI Option-like struct with `has_value` and `value` fields
 * to a nonstd::optional.
 */
template <typename OptionLike>
auto to_optional(OptionLike e) {
    return e.has_value ? nonstd::make_optional(e.value) : nonstd::nullopt;
}

/**
 * @brief Converts an initializer_list to a span.
 */
template <typename T>
nonstd::span<const T> to_span(std::initializer_list<T> list) {
    return {list.begin(), list.size()};
}

/**
 * @brief Converts the argument to `rust::Str`
 */
inline rust::Str to_rust_str(nonstd::string_view string_view) {
    return {string_view.data(), string_view.size()};
}

/**
 * @brief Converts the argument to `nonstd::string_view`
 */
inline nonstd::string_view to_string_view(rust::Str str) {
    return {str.data(), str.size()};
}

/**
 * @brief Converts the argument to `rust::String`
 */
inline rust::String to_rust_string(nonstd::string_view string_view) {
    return {string_view.data(), string_view.size()};
}

/**
 * @brief Converts the argument to `rust::String`
 */
inline rust::String to_rust_string(std::string string) { return string; }

/**
 * @brief Converts the argument to `rust::String`. This overload exists for use
 * in templates.
 */
inline rust::String to_rust_string(rust::String string) { return string; }

/**
 * @brief Converts the argument to `rust::String`
 */
inline rust::String to_rust_string(const char* ptr) { return ptr; }

/**
 * @brief Converts the argument to `rust::Slice`
 */
template <typename T>
inline rust::Slice<const T> to_rust_slice(nonstd::span<T> span) {
    return {span.data(), span.size()};
}

/**
 * @brief Converts the argument to `rust::Slice`
 */
template <typename T>
inline rust::Slice<const T> to_rust_slice(std::vector<T> vec) {
    return {vec.data(), vec.size()};
}

/**
 * @brief Converts the argument to `rust::Slice`
 */
template <typename T>
inline rust::Slice<const T> to_rust_slice(std::initializer_list<T> list) {
    return {list.begin(), list.size()};
}

/**
 * @brief Converts the argument to `rust::Slice`
 */
template <typename T>
inline rust::Slice<const T> to_rust_slice(rust::Vec<T> vec) {
    return {vec.data(), vec.size()};
}

/**
 * @brief Fills a vector with transformed data from another container
 *
 * @param vec The vector to fill (assumed to be empty initially)
 * @param cpp_container The source container
 * @param f The lambda to transform elements of cpp_container
 */
template <typename Vec, typename Container, typename F>
void fill_vec(Vec& vec, const Container& cpp_container, F f) {
    vec.reserve(cpp_container.size());
    for (auto x : cpp_container) {
        vec.push_back(f(x));
    }
}

/**
 * @brief Fills a vector with data from another container (like fill_vec(Vec&, const Container&, F) but with identity function).
 *
 * @param vec The vector to fill (assumed to be empty initially)
 * @param cpp_container The source container
 */
template <typename Vec, typename Container>
void fill_vec(Vec& vec, const Container& cpp_container) {
    fill_vec(vec, cpp_container, [](auto x) { return x; });
}

template <class F>
void vararg_for(F f) {}

template <class F, class Arg, class... Args>
void vararg_for(F f, Arg arg, Args... args) {
    f(std::forward<Arg>(arg));
    vararg_for(f, std::forward<Args>(args)...);
}
}
}  // namespace tokenizers
}  // namespace huggingface
