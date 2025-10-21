#pragma once
#include <string>
#include <vector>
#include <cstdint>

extern "C" {
    struct tokenizers_encoding_t {
        const int32_t* ids;
        size_t len;
    };

    void* tokenizers_new_from_file(const char* path);
    void* tokenizers_new_from_str(const char* json);
    void tokenizers_free(void* tokenizer);
    tokenizers_encoding_t tokenizers_encode(void* tokenizer, const char* text, bool add_special_tokens);
    void tokenizers_free_encoding(tokenizers_encoding_t enc);
    const char* tokenizers_version();
    void tokenizers_string_free(char* s);
    size_t tokenizers_vocab_size(void* tokenizer);
    int32_t tokenizers_token_to_id(void* tokenizer, const char* token);
    bool tokenizers_add_special_token(void* tokenizer, const char* token);
}

namespace tokenizers {

class Tokenizer {
public:
    Tokenizer() = default;
    explicit Tokenizer(const std::string& path) { load(path); }
    ~Tokenizer() { reset(); }
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;
    Tokenizer(Tokenizer&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
    Tokenizer& operator=(Tokenizer&& other) noexcept {
        if (this != &other) {
            reset();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    bool load(const std::string& path) {
        reset();
        handle_ = tokenizers_new_from_file(path.c_str());
        return handle_ != nullptr;
    }

    std::vector<int32_t> encode(const std::string& text, bool add_special_tokens = true) const {
        if (!handle_) return {};
        tokenizers_encoding_t enc = tokenizers_encode(handle_, text.c_str(), add_special_tokens);
        std::vector<int32_t> out;
        if (enc.ids && enc.len) {
            out.assign(enc.ids, enc.ids + enc.len);
        }
        tokenizers_free_encoding(enc);
        return out;
    }

    size_t vocab_size() const {
        if (!handle_) return 0;
        return tokenizers_vocab_size(handle_);
    }

    int32_t token_to_id(const std::string& token) const {
        if (!handle_) return -1;
        return tokenizers_token_to_id(handle_, token.c_str());
    }

    bool add_special_token(const std::string& token) {
        if (!handle_) return false;
        return tokenizers_add_special_token(handle_, token.c_str());
    }

    bool valid() const { return handle_ != nullptr; }

    static std::string version() {
        const char* v = tokenizers_version();
        if (!v) return {};
        std::string s(v);
        // version string is allocated, free if not static; current impl returns dynamic
        tokenizers_string_free(const_cast<char*>(v));
        return s;
    }

private:
    void reset() {
        if (handle_) {
            tokenizers_free(handle_);
            handle_ = nullptr;
        }
    }

    void* handle_ = nullptr;
};

} // namespace tokenizers
