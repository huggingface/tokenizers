#pragma once

#define BUILDER_ARG(type, name, default_value) \
    type name = default_value;                 \
    auto& with_##name(type name) {             \
        this->name = name;                     \
        return *this;                          \
    }

#define DELETE_COPY(type)       \
    type(const type&) = delete; \
    type& operator=(const type&) = delete;
