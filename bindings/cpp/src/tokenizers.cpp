/**
 * Tokenizer C++ bindings implementation
 */

#include <tokenizers/tokenizers.h>
#include <jinja2cpp/template.h>
#include <jinja2cpp/value.h>

namespace tokenizers {

std::string Tokenizer::apply_chat_template(
    const std::vector<ChatMessage>& messages,
    bool add_generation_prompt
) const {
    // Get the template string
    std::string tmpl_str = chat_template();
    if (tmpl_str.empty()) {
        throw ChatTemplateError("No chat template available for this tokenizer");
    }
    
    // Create Jinja2 template
    jinja2::Template tpl;
    auto load_result = tpl.Load(tmpl_str, "chat_template");
    if (!load_result) {
        throw ChatTemplateError("Failed to parse chat template: " + 
            load_result.error().ToString());
    }
    
    // Convert messages to Jinja2 values
    jinja2::ValuesList jinja_messages;
    for (const auto& msg : messages) {
        jinja2::ValuesMap msg_map;
        msg_map["role"] = msg.role;
        msg_map["content"] = msg.content;
        jinja_messages.push_back(std::move(msg_map));
    }
    
    // Build parameters map
    jinja2::ValuesMap params;
    params["messages"] = std::move(jinja_messages);
    params["add_generation_prompt"] = add_generation_prompt;
    
    // Add special tokens as variables (commonly used in templates)
    params["bos_token"] = bos_token();
    params["eos_token"] = eos_token();
    params["pad_token"] = pad_token();
    params["unk_token"] = unk_token();
    
    // Render the template
    auto render_result = tpl.RenderAsString(params);
    if (!render_result) {
        throw ChatTemplateError("Failed to render chat template: " + 
            render_result.error().ToString());
    }
    
    return render_result.value();
}

} // namespace tokenizers
