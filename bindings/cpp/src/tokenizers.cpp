/**
 * Tokenizer C++ bindings implementation
 */

#include <tokenizers/tokenizers.h>
#include <sstream>
#include <iomanip>

namespace tokenizers {

// Helper to escape JSON strings - handles special characters properly
static std::string json_escape(const std::string& input) {
    std::string output;
    output.reserve(input.size() * 1.1);  // Reserve extra space for escapes
    for (unsigned char c : input) {
        switch (c) {
            case '"': output += "\\\""; break;
            case '\\': output += "\\\\"; break;
            case '\b': output += "\\b"; break;
            case '\f': output += "\\f"; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default:
                if (c < 0x20) {
                    // Control characters: escape as \uXXXX
                    char buf[7];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    output += buf;
                } else {
                    output += c;
                }
        }
    }
    return output;
}

std::string Tokenizer::apply_chat_template(
    const std::string& template_str,
    const std::vector<ChatMessage>& messages,
    bool add_generation_prompt
) const {
    // Build messages JSON array manually
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < messages.size(); ++i) {
        if (i > 0) ss << ",";
        ss << "{\"role\":\"" << json_escape(messages[i].role) 
           << "\",\"content\":\"" << json_escape(messages[i].content) << "\"}";
    }
    ss << "]";
    std::string messages_json_str = ss.str();
    
    // Get special tokens (pass as C strings, can be null)
    std::string bos_str = bos_token();
    std::string eos_str = eos_token();
    const char* bos_ptr = bos_str.empty() ? nullptr : bos_str.c_str();
    const char* eos_ptr = eos_str.empty() ? nullptr : eos_str.c_str();
    
    // Call C FFI function with custom template
    char* error_msg = nullptr;
    char* result = tokenizers_apply_chat_template(
        handle_,
        template_str.c_str(),
        messages_json_str.c_str(),
        add_generation_prompt,
        bos_ptr,
        eos_ptr,
        &error_msg
    );
    
    if (result == nullptr) {
        std::string error = error_msg ? error_msg : "Failed to apply chat template";
        if (error_msg) {
            tokenizers_string_free(error_msg);
        }
        throw ChatTemplateError(error);
    }
    
    std::string rendered(result);
    tokenizers_string_free(result);
    
    return rendered;
}

std::string Tokenizer::apply_chat_template(
    const std::vector<ChatMessage>& messages,
    bool add_generation_prompt
) const {
    // Get the template string from config and delegate to the overload
    std::string tmpl_str = chat_template();
    if (tmpl_str.empty()) {
        throw ChatTemplateError("No chat template available for this tokenizer");
    }
    return apply_chat_template(tmpl_str, messages, add_generation_prompt);
}

} // namespace tokenizers
