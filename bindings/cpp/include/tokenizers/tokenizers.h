#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>

// Forward declare jinja2 types to avoid pulling in heavy headers
namespace jinja2 { class Template; }

extern "C" {
    struct tokenizers_encoding_t {
        const int32_t* ids;
        const int32_t* attention_mask;
        size_t len;
        void* _internal_ptr;  // Internal use only - do not access
    };

    struct tokenizers_padding_params_t {
        uint32_t pad_id;
        uint32_t pad_type_id;
        const char* pad_token;
        int strategy;
        size_t fixed_length;
        int direction;
        size_t pad_to_multiple_of;
    };

    struct tokenizers_truncation_params_t {
        size_t max_length;
        size_t stride;
        int strategy;
        int direction;
    };

    void* tokenizers_new_from_file(const char* path);
    void* tokenizers_new_from_file_with_config(const char* path, const char* config_path);
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
    size_t tokenizers_add_tokens(void* tokenizer, const char** tokens, size_t len);
    tokenizers_encoding_t* tokenizers_encode_batch(void* tokenizer, const char** texts, size_t len, bool add_special_tokens);
    void tokenizers_free_batch_encoding(tokenizers_encoding_t* encodings, size_t len);
    char** tokenizers_decode_batch(void* tokenizer, const int32_t** ids, const size_t* lens, size_t batch_len, bool skip_special_tokens);
    void tokenizers_free_batch_decode(char** strings, size_t len);
    void tokenizers_set_padding(void* tokenizer, const tokenizers_padding_params_t* params);
    void tokenizers_set_truncation(void* tokenizer, const tokenizers_truncation_params_t* params);
    
    // Unified special token API (auto-uses config if available, falls back to heuristic)
    int32_t tokenizers_get_special_token_id(void* tokenizer, const char* name);
    char* tokenizers_get_special_token(void* tokenizer, const char* name);
    bool tokenizers_get_add_bos_token(void* tokenizer);
    bool tokenizers_get_add_eos_token(void* tokenizer);
    bool tokenizers_has_chat_template(void* tokenizer);
    char* tokenizers_get_chat_template(void* tokenizer);
    char* tokenizers_apply_chat_template(
        void* tokenizer,
        const char* template_str,
        const char* messages_json,
        bool add_generation_prompt,
        const char* bos_token,
        const char* eos_token,
        char** error_out
    );
}

namespace tokenizers {

struct Encoding {
    std::vector<int32_t> ids;
    std::vector<int32_t> attention_mask;
    
    operator std::vector<int32_t>() const { return ids; }

    size_t size() const { return ids.size(); }
    bool empty() const { return ids.empty(); }
    int32_t operator[](size_t i) const { return ids[i]; }
    std::vector<int32_t>::const_iterator begin() const { return ids.begin(); }
    std::vector<int32_t>::const_iterator end() const { return ids.end(); }
    
    bool operator==(const Encoding& other) const {
        return ids == other.ids && attention_mask == other.attention_mask;
    }
    bool operator!=(const Encoding& other) const {
        return !(*this == other);
    }
};

/// Chat message for apply_chat_template
struct ChatMessage {
    std::string role;     // "system", "user", "assistant"
    std::string content;  // Message content
};

/// Exception for chat template errors
class ChatTemplateError : public std::runtime_error {
public:
    explicit ChatTemplateError(const std::string& msg) : std::runtime_error(msg) {}
};

struct PaddingParams {
    uint32_t pad_id = 0;
    uint32_t pad_type_id = 0;
    std::string pad_token = "[PAD]";
    enum Strategy { BatchLongest = 0, Fixed = 1 } strategy = BatchLongest;
    size_t fixed_length = 0;
    enum Direction { Left = 0, Right = 1 } direction = Right;
    size_t pad_to_multiple_of = 0;
};

struct TruncationParams {
    size_t max_length = 512;
    size_t stride = 0;
    enum Strategy { LongestFirst = 0, OnlyFirst = 1, OnlySecond = 2 } strategy = LongestFirst;
    enum Direction { Left = 0, Right = 1 } direction = Right;
};

class Tokenizer {
public:
    Tokenizer() = default;
    /// Load tokenizer from file, auto-loads tokenizer_config.json if present
    explicit Tokenizer(const std::string& path) { load(path); }
    /// Load tokenizer with explicit config file path
    Tokenizer(const std::string& path, const std::string& config_path) { load(path, config_path); }
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

    /// Load tokenizer, auto-loads tokenizer_config.json if present
    bool load(const std::string& path) {
        reset();
        handle_ = tokenizers_new_from_file(path.c_str());
        return handle_ != nullptr;
    }
    
    /// Load tokenizer with explicit config file path
    bool load(const std::string& path, const std::string& config_path) {
        reset();
        handle_ = tokenizers_new_from_file_with_config(path.c_str(), config_path.c_str());
        return handle_ != nullptr;
    }

    Encoding encode(const std::string& text, bool add_special_tokens = true) const {
        if (!handle_) return {};
        tokenizers_encoding_t enc = tokenizers_encode(handle_, text.c_str(), add_special_tokens);
        Encoding out;
        if (enc.ids && enc.len) {
            out.ids.assign(enc.ids, enc.ids + enc.len);
        }
        if (enc.attention_mask && enc.len) {
            out.attention_mask.assign(enc.attention_mask, enc.attention_mask + enc.len);
        }
        tokenizers_free_encoding(enc);
        return out;
    }

    std::vector<Encoding> encode_batch(const std::vector<std::string>& texts, bool add_special_tokens = true) const {
        if (!handle_) return {};
        std::vector<const char*> c_texts;
        c_texts.reserve(texts.size());
        for (const auto& t : texts) c_texts.push_back(t.c_str());
        
        tokenizers_encoding_t* encs = tokenizers_encode_batch(handle_, c_texts.data(), c_texts.size(), add_special_tokens);
        if (!encs) return {};
        
        std::vector<Encoding> out;
        out.reserve(texts.size());
        for (size_t i = 0; i < texts.size(); ++i) {
            Encoding e;
            if (encs[i].ids && encs[i].len) {
                e.ids.assign(encs[i].ids, encs[i].ids + encs[i].len);
            }
            if (encs[i].attention_mask && encs[i].len) {
                e.attention_mask.assign(encs[i].attention_mask, encs[i].attention_mask + encs[i].len);
            }
            out.push_back(std::move(e));
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

    std::vector<std::string> decode_batch(const std::vector<std::vector<int32_t>>& batch_ids, bool skip_special_tokens = true) const {
        if (!handle_) return {};
        std::vector<const int32_t*> c_ids;
        std::vector<size_t> c_lens;
        c_ids.reserve(batch_ids.size());
        c_lens.reserve(batch_ids.size());
        
        for (const auto& ids : batch_ids) {
            c_ids.push_back(ids.data());
            c_lens.push_back(ids.size());
        }
        
        char** strings = tokenizers_decode_batch(handle_, c_ids.data(), c_lens.data(), batch_ids.size(), skip_special_tokens);
        if (!strings) return {};
        
        std::vector<std::string> res;
        res.reserve(batch_ids.size());
        for (size_t i = 0; i < batch_ids.size(); ++i) {
            if (strings[i]) {
                res.emplace_back(strings[i]);
            } else {
                res.emplace_back("");
            }
        }
        tokenizers_free_batch_decode(strings, batch_ids.size());
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

    void set_padding(const PaddingParams& params) {
        if (!handle_) return;
        tokenizers_padding_params_t c_params;
        c_params.pad_id = params.pad_id;
        c_params.pad_type_id = params.pad_type_id;
        c_params.pad_token = params.pad_token.c_str();
        c_params.strategy = (int)params.strategy;
        c_params.fixed_length = params.fixed_length;
        c_params.direction = (int)params.direction;
        c_params.pad_to_multiple_of = params.pad_to_multiple_of;
        
        tokenizers_set_padding(handle_, &c_params);
    }
    
    void disable_padding() {
        if (!handle_) return;
        tokenizers_set_padding(handle_, nullptr);
    }

    void set_truncation(const TruncationParams& params) {
        if (!handle_) return;
        tokenizers_truncation_params_t c_params;
        c_params.max_length = params.max_length;
        c_params.stride = params.stride;
        c_params.strategy = (int)params.strategy;
        c_params.direction = (int)params.direction;
        
        tokenizers_set_truncation(handle_, &c_params);
    }

    void disable_truncation() {
        if (!handle_) return;
        tokenizers_set_truncation(handle_, nullptr);
    }

    size_t add_tokens(const std::vector<std::string>& tokens) {
        if (!handle_) return 0;
        std::vector<const char*> c_tokens;
        c_tokens.reserve(tokens.size());
        for (const auto& t : tokens) c_tokens.push_back(t.c_str());
        return tokenizers_add_tokens(handle_, c_tokens.data(), c_tokens.size());
    }

    // === Special Token API (unified - auto-uses config if available) ===
    
    /// Get special token ID by name ("BOS", "EOS", "PAD", "UNK")
    /// Auto-uses tokenizer_config.json if present, falls back to heuristic.
    int32_t special_token_id(const std::string& name) const {
        if (!handle_) return -1;
        return tokenizers_get_special_token_id(handle_, name.c_str());
    }
    
    /// Get special token string by name ("BOS", "EOS", "PAD", "UNK")
    /// Returns token from config if available, empty string otherwise.
    std::string special_token(const std::string& name) const {
        if (!handle_) return {};
        char* s = tokenizers_get_special_token(handle_, name.c_str());
        if (!s) return {};
        std::string res(s);
        tokenizers_string_free(s);
        return res;
    }
    
    // Convenience ID accessors
    int32_t bos_id() const { return special_token_id("BOS"); }
    int32_t eos_id() const { return special_token_id("EOS"); }
    int32_t pad_id() const { return special_token_id("PAD"); }
    int32_t unk_id() const { return special_token_id("UNK"); }
    
    // Convenience token string accessors
    std::string bos_token() const { return special_token("BOS"); }
    std::string eos_token() const { return special_token("EOS"); }
    std::string pad_token() const { return special_token("PAD"); }
    std::string unk_token() const { return special_token("UNK"); }
    
    /// Whether config specifies BOS token should be added
    bool add_bos_token() const {
        if (!handle_) return false;
        return tokenizers_get_add_bos_token(handle_);
    }
    
    /// Whether config specifies EOS token should be added  
    bool add_eos_token() const {
        if (!handle_) return false;
        return tokenizers_get_add_eos_token(handle_);
    }
    
    /// Check if tokenizer has a chat template (from config)
    bool has_chat_template() const {
        if (!handle_) return false;
        return tokenizers_has_chat_template(handle_);
    }
    
    /// Get the raw chat template string (Jinja2 template)
    std::string chat_template() const {
        if (!handle_) return {};
        char* s = tokenizers_get_chat_template(handle_);
        if (!s) return {};
        std::string res(s);
        tokenizers_string_free(s);
        return res;
    }
    
    /// Apply chat template to format messages
    /// @param messages Vector of ChatMessage with role and content
    /// @param add_generation_prompt If true, adds prompt for assistant response
    /// @return Formatted string ready for tokenization
    /// @throws ChatTemplateError if no template or rendering fails
    std::string apply_chat_template(
        const std::vector<ChatMessage>& messages,
        bool add_generation_prompt = true
    ) const;

    /// Apply custom chat template to messages
    /// @param template_str The Jinja2 chat template string to use
    /// @param messages Vector of ChatMessage with role and content
    /// @param add_generation_prompt If true, adds prompt for assistant response
    /// @return Formatted string ready for tokenization
    /// @throws ChatTemplateError if template rendering fails
    std::string apply_chat_template(
        const std::string& template_str,
        const std::vector<ChatMessage>& messages,
        bool add_generation_prompt = true
    ) const;

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
