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
enum class InputSequenceTag : uint8_t { Str, String, StringSlice };

class InputSequence {
private:
    // this could be (non)std::variant, but I don't want to add the
    // dependency just for a single use
    enum { STR, STRING, STRING_VECTOR } tag_;
    union {
        rust::Str str_;
        rust::String string_;
        std::vector<rust::String> string_vector_;
    };

public:
    InputSequence() = delete;
    InputSequence(const InputSequence& other) : tag_(other.tag_) {
        switch (tag_) {
            case STR:
                str_ = other.str_;
                break;
            case STRING:
                string_ = other.string_;
                break;
            case STRING_VECTOR:
                string_vector_ = other.string_vector_;
                break;
            default:
                // we really shouldn't get here!
                assert(false &&
                       "a tag was not covered in the copy constructor");
        }
    };
    InputSequence(nonstd::string_view str)
        : tag_(STR), str_(ffi::to_rust_str(str)){};
    InputSequence(std::string&& str)
        : tag_(STRING), string_(ffi::to_rust_string(str)){};
    InputSequence(rust::String&& str) : tag_(STRING), string_(str){};
    InputSequence(const char* str) : InputSequence(nonstd::string_view(str)){};
    InputSequence(nonstd::span<std::string> strs)
        : tag_(STRING_VECTOR), string_vector_() {
        ffi::fill_vec(string_vector_, strs,
                      [](auto x) { return rust::String(x); });
    };
    InputSequence(std::initializer_list<rust::String> strs)
        : tag_(STRING_VECTOR), string_vector_() {
        ffi::fill_vec(string_vector_, strs);
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
            case STRING_VECTOR:
                string_vector_.~vector();
                break;
            default:
                // we really shouldn't get here!
                assert(false && "a tag was not covered in the destructor");
        }
    }

    InputSequenceTag get_tag() const {
        switch (tag_) {
            case STR:
            case STRING:
                return InputSequenceTag::Str;
            case STRING_VECTOR:
                return InputSequenceTag::StringSlice;
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

    rust::Slice<const rust::String> get_string_slice() const {
        switch (tag_) {
            case STRING_VECTOR:
                return {string_vector_.data(), string_vector_.size()};
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
