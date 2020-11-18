#pragma once

#define BUILDER_ARG(type, name, default_value)                        \
    type name = default_value;                                        \
    _Pragma("warning(suppress: 4458)") auto& with_##name(type name) { \
        this->name = name;                                            \
        return *this;                                                 \
    }

// Ideally we want inner_ to be private, but I couldn't make it compile
#define FFI_WRAPPER_MEMBERS(type)                                              \
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