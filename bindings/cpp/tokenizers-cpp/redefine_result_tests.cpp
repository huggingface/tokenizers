#include <cstddef>

template <typename T>
struct fake_result {
    static fake_result<T> success(T&&) { return {}; }
    static fake_result<T> error(const char*) { return {}; }
};

#define HFT_RESULT_VOID fake_result<std::nullptr_t>
#define HFT_RESULT(type) fake_result<type>

#define HFT_TRY_VOID(function_call)                           \
    try {                                                     \
        function_call;                                        \
        return fake_result<std::nullptr_t>::success(nullptr); \
    } catch (const std::exception& e) {                       \
        return fake_result<std::nullptr_t>::error(e.what());  \
    }

#define HFT_TRY(T, function_call)                      \
    try {                                              \
        return fake_result<T>::success(function_call); \
    } catch (const std::exception& e) {                \
        return fake_result<T>::error(e.what());        \
    }

#include "tokenizers-cpp/normalizers.h"
#include "tokenizers-cpp/pre_tokenizers.h"
