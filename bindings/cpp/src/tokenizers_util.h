#pragma once

#define BUILDER_ARG(type, name, default_value) \
private:                                       \
    type name##_ = default_value;              \
                                               \
public:                                        \
    auto& name(type name) {                    \
        this->name##_ = name;                  \
        return *this;                          \
    }

#define DELETE_COPY(type)       \
    type(const type&) = delete; \
    type& operator=(const type&) = delete;
