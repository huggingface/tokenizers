#pragma once

#include "tokenizers-cpp/common.h"

#include "rust/cxx.h"
#include <nonstd/span.hpp>
#include <cassert>

namespace huggingface {
namespace tokenizers {
// can't use these at the moment:
// StrSlice, &[&str] not supported by cxx
// StrVec, Vec<&str> not supported by cxx
// StrVec in particular will be useful
enum class InputSequenceTag : uint8_t { Str, String, StringVec, StringSlice };

class InputSequence {
private:
    // this could be (non)std::variant, but I don't want to add the
    // dependency just for a single use
    enum { STR, STRING, STRING_VEC } tag_;
    union {
        rust::Str str_;
        rust::String string_;
        rust::Vec<rust::String> string_vec_;
    };

public:
    InputSequence() = delete;
    InputSequence(nonstd::string_view str)
        : tag_(STR), str_(ffi::to_rust_str(str)){};
    InputSequence(const char* str) : InputSequence(nonstd::string_view(str)){};
    InputSequence(nonstd::span<std::string> strs)
        : tag_(STRING_VEC), string_vec_() {
        ffi::fill_vec(string_vec_, strs,
                      [](auto x) { return rust::String(x); });
    };
    // TODO add other constructors

    ~InputSequence() noexcept {
        switch (tag_) {
            case STR:
                str_.~Str();
                break;
            case STRING:
                string_.~String();
                break;
            case STRING_VEC:
                string_vec_.~Vec();
                break;
            default:
                // we really shouldn't get here!
                assert(false && "a tag was not covered in the destructor");
        }
    }

    InputSequenceTag get_tag() const {
        switch (tag_) {
            case STR:
                return InputSequenceTag::Str;
            case STRING:
                return InputSequenceTag::Str;
            case STRING_VEC:
                return InputSequenceTag::StringVec;
            default:
                throw std::logic_error("invalid tag");
        }
    }

    rust::Str get_str() const {
        switch (tag_) {
            case STR:
                return str_;
            case STRING:
                return rust::Str(string_);
            default:
                throw std::logic_error("wrong tag to return a &str");
        }
    }

    rust::String get_string() const {
        switch (tag_) {
            case STRING:
                return string_;
            default:
                throw std::logic_error("wrong tag to return a String");
        }
    }

    rust::Vec<rust::String> get_string_vec() const {
        throw std::logic_error("wrong tag to return a Vec<String>");
    }

    rust::Slice<const rust::String> get_string_slice() const {
        switch (tag_) {
            case STRING_VEC:
                return {string_vec_.data(), string_vec_.size()};
            default:
                throw std::logic_error("wrong tag to return a [String]");
        }
    }
};

struct InputSequencePair {
    InputSequence first_;
    InputSequence second_;

    const InputSequence& first() const { return first_; }

    const InputSequence& second() const { return second_; }
};
}  // namespace tokenizers
}  // namespace huggingface
