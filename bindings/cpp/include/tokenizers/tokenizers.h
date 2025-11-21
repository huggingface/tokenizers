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
    char* tokenizers_id_to_token(void* tokenizer, int32_t id);
    char* tokenizers_decode(void* tokenizer, const int32_t* ids, size_t len, bool skip_special_tokens);
    bool tokenizers_save(void* tokenizer, const char* path, bool pretty);
    char* tokenizers_to_str(void* tokenizer, bool pretty);
    bool tokenizers_add_special_token(void* tokenizer, const char* token);
    size_t tokenizers_add_special_tokens(void* tokenizer, const char** tokens, size_t len);
    tokenizers_encoding_t* tokenizers_encode_batch(void* tokenizer, const char** texts, size_t len, bool add_special_tokens);
    void tokenizers_free_batch_encoding(tokenizers_encoding_t* encodings, size_t len);
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

    static Tokenizer FromBlobJSON(const std::string& json) {
        Tokenizer t;
        t.handle_ = tokenizers_new_from_str(json.c_str());
        return t;
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

    std::vector<std::vector<int32_t>> encode_batch(const std::vector<std::string>& texts, bool add_special_tokens = true) const {
        if (!handle_) return {};
        std::vector<const char*> c_texts;
        c_texts.reserve(texts.size());
        for (const auto& t : texts) c_texts.push_back(t.c_str());
        
        tokenizers_encoding_t* encs = tokenizers_encode_batch(handle_, c_texts.data(), c_texts.size(), add_special_tokens);
        if (!encs) return {};
        
        std::vector<std::vector<int32_t>> out;
        out.reserve(texts.size());
        for (size_t i = 0; i < texts.size(); ++i) {
            std::vector<int32_t> ids;
            if (encs[i].ids && encs[i].len) {
                ids.assign(encs[i].ids, encs[i].ids + encs[i].len);
            }
            out.push_back(std::move(ids));
        }
        tokenizers_free_batch_encoding(encs, texts.size());
        return out;
    }

    std::string decode(const std::vector<int32_t>& ids, bool skip_special_tokens = true) const {
        if (!handle_) return {};
        char* s = tokenizers_decode(handle_, ids.data(), ids.size(), skip_special_tokens);
        if (!s) return {};
        std::string res(s);
        tokenizers_string_free(s);
        return res;
    }

    size_t vocab_size() const {
        if (!handle_) return 0;
        return tokenizers_vocab_size(handle_);
    }

    int32_t token_to_id(const std::string& token) const {
        if (!handle_) return -1;
        return tokenizers_token_to_id(handle_, token.c_str());
    }

    std::string id_to_token(int32_t id) const {
        if (!handle_) return {};
        char* s = tokenizers_id_to_token(handle_, id);
        if (!s) return {};
        std::string res(s);
        tokenizers_string_free(s);
        return res;
    }

    bool save(const std::string& path, bool pretty = true) const {
        if (!handle_) return false;
        return tokenizers_save(handle_, path.c_str(), pretty);
    }

    std::string to_string(bool pretty = false) const {
        if (!handle_) return {};
        char* s = tokenizers_to_str(handle_, pretty);
        if (!s) return {};
        std::string res(s);
        tokenizers_string_free(s);
        return res;
    }

    bool add_special_token(const std::string& token) {
        if (!handle_) return false;
        return tokenizers_add_special_token(handle_, token.c_str());
    }

    size_t add_special_tokens(const std::vector<std::string>& tokens) {
        if (!handle_) return 0;
        std::vector<const char*> c_tokens;
        c_tokens.reserve(tokens.size());
        for (const auto& t : tokens) c_tokens.push_back(t.c_str());
        return tokenizers_add_special_tokens(handle_, c_tokens.data(), c_tokens.size());
    }

    bool valid() const { return handle_ != nullptr; }

    static std::string version() {
        const char* v = tokenizers_version();
        if (!v) return {};
        std::string s(v);
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
