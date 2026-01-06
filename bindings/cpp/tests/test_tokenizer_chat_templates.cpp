/**
 * Chat Template Tests for Tokenizer C++ bindings
 * 
 * Tests chat template functionality using custom templates and tokenizer configurations.
 * These tests verify that custom chat templates can be applied through the C++ bindings.
 */
#include <gtest/gtest.h>
#include <tokenizers/tokenizers.h>
#include "test_common.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <map>

using namespace tokenizers;
using test_utils::find_resource;

// ==================== Template Loading ====================

std::map<std::string, std::string> load_templates() {
    std::map<std::string, std::string> templates;
    std::string test_dir = std::string(__FILE__);
    size_t last_slash = test_dir.find_last_of("/\\");
    test_dir = test_dir.substr(0, last_slash);
    std::string template_file = test_dir + "/chat-template-tests.txt";
    
    std::ifstream file(template_file);
    if (!file.is_open()) {
        // Return empty map on error - caller will check
        return templates;
    }
    
    std::string line;
    std::string current_name;
    std::string current_template;
    
    while (std::getline(file, line)) {
        // Check for template name line
        if (line.find("TEMPLATE:") != std::string::npos) {
            // Save previous template if exists
            if (!current_name.empty()) {
                templates[current_name] = current_template;
            }
            
            // Extract template name
            size_t pos = line.find("TEMPLATE:") + 9;
            current_name = line.substr(pos);
            // Trim whitespace
            current_name.erase(0, current_name.find_first_not_of(" \t"));
            current_name.erase(current_name.find_last_not_of(" \t") + 1);
            current_template = "";
        }
        // Check for delimiter
        else if (line.find("==##==") != std::string::npos) {
            // Skip delimiter lines
            continue;
        }
        // Add to current template
        else if (!current_name.empty()) {
            if (!current_template.empty()) {
                current_template += "\n";
            }
            current_template += line;
        }
    }
    
    // Save last template
    if (!current_name.empty()) {
        templates[current_name] = current_template;
    }
    
    return templates;
}

// ==================== Chat Template Tests ====================

class CustomChatTemplateTest : public ::testing::Test {
protected:
    std::map<std::string, std::string> templates;
    
    void SetUp() override {
        templates = load_templates();
        if (templates.empty()) {
            GTEST_SKIP() << "No templates loaded from chat-template-tests.txt";
        }
    }
};

TEST_F(CustomChatTemplateTest, BasicMarkdownTemplate) {
    std::vector<ChatMessage> messages = {
        {"user", "Hello!"},
        {"assistant", "Hi there!"},
        {"user", "How are you?"}
    };
    
    std::string result = Tokenizer(find_resource("tokenizer.json")).apply_chat_template(
        templates["BasicMarkdown"],
        messages,
        true
    );
    
    // Template should include the content with markdown formatting
    EXPECT_NE(result.find("### User:"), std::string::npos);
    EXPECT_NE(result.find("### Assistant:"), std::string::npos);
    EXPECT_NE(result.find("Hello!"), std::string::npos);
    EXPECT_NE(result.find("How are you?"), std::string::npos);
}

TEST_F(CustomChatTemplateTest, LlamaStyleTemplate) {
    std::vector<ChatMessage> messages = {
        {"user", "What is AI?"},
        {"assistant", "Artificial Intelligence"}
    };
    
    std::string result = Tokenizer(find_resource("tokenizer.json")).apply_chat_template(
        templates["LlamaStyle"],
        messages,
        false
    );
    
    // Template should use [USER] and [ASSISTANT] markers
    EXPECT_NE(result.find("[USER]"), std::string::npos);
    EXPECT_NE(result.find("[ASSISTANT]"), std::string::npos);
    EXPECT_NE(result.find("What is AI?"), std::string::npos);
}

TEST_F(CustomChatTemplateTest, StrictAlternatingTemplate) {
    std::vector<ChatMessage> messages = {
        {"user", "First question"},
        {"assistant", "First answer"},
        {"user", "Second question"}
    };
    
    std::string result = Tokenizer(find_resource("tokenizer.json")).apply_chat_template(
        templates["StrictAlternating"],
        messages,
        true
    );
    
    // Template should use <USER> and <ASSISTANT> tags
    EXPECT_NE(result.find("<USER>"), std::string::npos);
    EXPECT_NE(result.find("<ASSISTANT>"), std::string::npos);
    EXPECT_NE(result.find("First question"), std::string::npos);
    EXPECT_NE(result.find("First answer"), std::string::npos);
}

TEST_F(CustomChatTemplateTest, CompactJsonStyleTemplate) {
    std::vector<ChatMessage> messages = {
        {"user", "Hello"},
        {"assistant", "Hi"}
    };
    
    std::string result = Tokenizer(find_resource("tokenizer.json")).apply_chat_template(
        templates["CompactJsonStyle"],
        messages,
        false
    );
    
    // Template should output JSON-like format
    EXPECT_NE(result.find("\"role\""), std::string::npos);
    EXPECT_NE(result.find("\"content\""), std::string::npos);
    EXPECT_NE(result.find("user"), std::string::npos);
    EXPECT_NE(result.find("assistant"), std::string::npos);
}

TEST_F(CustomChatTemplateTest, TagWithNewlinesTemplate) {
    std::vector<ChatMessage> messages = {
        {"system", "You are helpful"},
        {"user", "Help me"},
        {"assistant", "Of course"}
    };
    
    std::string result = Tokenizer(find_resource("tokenizer.json")).apply_chat_template(
        templates["TagWithNewlines"],
        messages,
        false
    );
    
    // Template should include SYSTEM, USER, ASSISTANT tags
    EXPECT_NE(result.find("SYSTEM:"), std::string::npos);
    EXPECT_NE(result.find("USER:"), std::string::npos);
    EXPECT_NE(result.find("ASSISTANT:"), std::string::npos);
    EXPECT_NE(result.find("---"), std::string::npos);
}

TEST_F(CustomChatTemplateTest, PrefixSuffixTemplate) {
    std::vector<ChatMessage> messages = {
        {"user", "Question"},
        {"assistant", "Answer"}
    };
    
    std::string result = Tokenizer(find_resource("tokenizer.json")).apply_chat_template(
        templates["PrefixSuffix"],
        messages,
        true
    );
    
    // Template should have BEGIN/END markers
    EXPECT_NE(result.find("[BEGIN_CONVERSATION]"), std::string::npos);
    EXPECT_NE(result.find("[END_CONVERSATION]"), std::string::npos);
    EXPECT_NE(result.find("[USER_START]"), std::string::npos);
    EXPECT_NE(result.find("[USER_END]"), std::string::npos);
}

TEST_F(CustomChatTemplateTest, MinimalWithDelimitersTemplate) {
    std::vector<ChatMessage> messages = {
        {"user", "Simple question"},
        {"assistant", "Simple answer"}
    };
    
    std::string result = Tokenizer(find_resource("tokenizer.json")).apply_chat_template(
        templates["MinimalWithDelimiters"],
        messages,
        false
    );
    
    // Minimal format with [role] prefix
    EXPECT_NE(result.find("[user]"), std::string::npos);
    EXPECT_NE(result.find("[assistant]"), std::string::npos);
    EXPECT_NE(result.find("Simple question"), std::string::npos);
}

TEST_F(CustomChatTemplateTest, HeadersWithContentTemplate) {
    std::vector<ChatMessage> messages = {
        {"user", "Test"},
        {"assistant", "Response"}
    };
    
    std::string result = Tokenizer(find_resource("tokenizer.json")).apply_chat_template(
        templates["HeadersWithContent"],
        messages,
        true
    );
    
    // Template should use <|im_start|> and <|im_end|> markers
    EXPECT_NE(result.find("<|im_start|>"), std::string::npos);
    EXPECT_NE(result.find("<|im_end|>"), std::string::npos);
}

TEST_F(CustomChatTemplateTest, ConditionalSystemMessageTemplate) {
    std::vector<ChatMessage> messages = {
        {"system", "Be concise"},
        {"user", "What?"},
        {"assistant", "Answer"}
    };
    
    std::string result = Tokenizer(find_resource("tokenizer.json")).apply_chat_template(
        templates["ConditionalSystemMessage"],
        messages,
        false
    );
    
    // Should conditionally handle system message
    EXPECT_NE(result.find("SYSTEM_MSG:"), std::string::npos);
    EXPECT_NE(result.find("user:"), std::string::npos);
}

TEST_F(CustomChatTemplateTest, WithRoleLabelTemplate) {
    std::vector<ChatMessage> messages = {
        {"user", "Question"},
        {"assistant", "Answer"}
    };
    
    std::string result = Tokenizer(find_resource("tokenizer.json")).apply_chat_template(
        templates["WithRoleLabel"],
        messages,
        true
    );
    
    // Template should use <<ROLE>> format
    EXPECT_NE(result.find("<<USER>>"), std::string::npos);
    EXPECT_NE(result.find("<<ASSISTANT>>"), std::string::npos);
    EXPECT_NE(result.find("<</USER>>"), std::string::npos);
    EXPECT_NE(result.find("<</ASSISTANT>>"), std::string::npos);
}

// ==================== Config Template Tests ====================

class ConfigChatTemplateTest : public ::testing::Test {
protected:
    Tokenizer tok;
    
    void SetUp() override {
        std::string path = find_resource("tokenizer.json");
        ASSERT_FALSE(path.empty()) << "Could not find tokenizer.json";
        tok = Tokenizer(path);
        ASSERT_TRUE(tok.valid());
    }
};

TEST_F(ConfigChatTemplateTest, ConfigTemplateExists) {
    // Verify the tokenizer has a chat template loaded from tokenizer_config.json
    EXPECT_TRUE(tok.has_chat_template());
    EXPECT_FALSE(tok.chat_template().empty());
}

TEST_F(ConfigChatTemplateTest, ApplyConfigTemplateBasic) {
    std::vector<ChatMessage> messages = {
        {"user", "Hello!"},
        {"assistant", "Hi there!"},
        {"user", "How are you?"}
    };
    
    std::string result = tok.apply_chat_template(messages, true);
    
    // Template should include the content
    EXPECT_NE(result.find("Hello!"), std::string::npos);
    EXPECT_NE(result.find("Hi there!"), std::string::npos);
    EXPECT_NE(result.find("How are you?"), std::string::npos);
}

TEST_F(ConfigChatTemplateTest, ConfigTemplateConsistency) {
    std::vector<ChatMessage> messages = {
        {"user", "Test message"},
        {"assistant", "Test response"}
    };
    
    std::string result1 = tok.apply_chat_template(messages, true);
    std::string result2 = tok.apply_chat_template(messages, true);
    
    // Same input should produce same output
    EXPECT_EQ(result1, result2);
}

TEST_F(ConfigChatTemplateTest, ConfigTemplateVsCustomTemplate) {
    std::vector<ChatMessage> messages = {
        {"user", "Same input"},
        {"assistant", "Different templates"}
    };
    
    std::string config_result = tok.apply_chat_template(messages, false);
    std::string custom_result = tok.apply_chat_template(
        tok.chat_template(),  // Use same template explicitly
        messages,
        false
    );
    
    // Should produce identical results
    EXPECT_EQ(config_result, custom_result);
}

// ==================== Error Handling Tests ====================

class ChatTemplateErrorTest : public ::testing::Test {
};

TEST_F(ChatTemplateErrorTest, NoChatTemplateError) {
    // BERT tokenizer typically doesn't have a chat template
    std::string path = find_resource("bert-wiki.json");
    if (path.empty()) {
        GTEST_SKIP() << "bert-wiki.json not found";
    }
    
    Tokenizer tok(path, "");  // Load without config
    ASSERT_TRUE(tok.valid());
    
    std::vector<ChatMessage> messages = {{"user", "test"}};
    
    // Should throw when no chat template available
    EXPECT_THROW(tok.apply_chat_template(messages), ChatTemplateError);
}

// ==================== Integration Tests ====================

class ChatTemplateIntegrationTest : public ::testing::Test {
protected:
    Tokenizer tok;
    std::map<std::string, std::string> templates;
    
    void SetUp() override {
        std::string path = find_resource("tokenizer.json");
        ASSERT_FALSE(path.empty()) << "Could not find tokenizer.json";
        tok = Tokenizer(path);
        ASSERT_TRUE(tok.valid());
        templates = load_templates();
        if (templates.empty()) {
            GTEST_SKIP() << "No templates loaded from chat-template-tests.txt";
        }
    }
};

TEST_F(ChatTemplateIntegrationTest, CustomTemplateAndTokenization) {
    // Verify custom template output can be tokenized
    std::vector<ChatMessage> messages = {
        {"user", "Hello"},
        {"assistant", "Hi"}
    };
    
    std::string formatted = tok.apply_chat_template(
        templates["BasicMarkdown"],
        messages,
        false
    );
    
    // Should be able to tokenize the formatted string
    auto encoding = tok.encode(formatted);
    EXPECT_FALSE(encoding.ids.empty());
    EXPECT_EQ(encoding.ids.size(), encoding.attention_mask.size());
}

TEST_F(ChatTemplateIntegrationTest, MultipleTemplatesProcessing) {
    std::vector<ChatMessage> messages = {
        {"user", "Test message"}
    };
    
    // All templates should process without errors
    for (const auto& [name, tmpl] : templates) {
        std::string result = tok.apply_chat_template(tmpl, messages, false);
        EXPECT_FALSE(result.empty()) << "Template " << name << " produced empty result";
        EXPECT_NE(result.find("Test message"), std::string::npos) 
            << "Template " << name << " didn't include message content";
    }
}

TEST_F(ChatTemplateIntegrationTest, LongConversationWithCustomTemplate) {
    // Test with a longer multi-turn conversation
    std::vector<ChatMessage> messages = {
        {"user", "What is AI?"},
        {"assistant", "AI is Artificial Intelligence"},
        {"user", "Tell me more"},
        {"assistant", "It uses algorithms and data"},
        {"user", "How does it learn?"},
        {"assistant", "Through training on data"},
        {"user", "What are applications?"},
        {"assistant", "Chat, image recognition, etc"},
        {"user", "Is it safe?"},
        {"assistant", "It depends on implementation"}
    };
    
    std::string formatted = tok.apply_chat_template(
        templates["LlamaStyle"],
        messages,
        true
    );
    
    // All messages should be in the output
    EXPECT_NE(formatted.find("What is AI?"), std::string::npos);
    EXPECT_NE(formatted.find("Tell me more"), std::string::npos);
    EXPECT_NE(formatted.find("Is it safe?"), std::string::npos);
}

TEST_F(ChatTemplateIntegrationTest, TemplateWithSpecialCharacters) {
    std::vector<ChatMessage> messages = {
        {"user", "Use \"quotes\", 'apostrophes', and\nline breaks"},
        {"assistant", "Response with: \\ backslash"}
    };
    
    std::string result = tok.apply_chat_template(
        templates["TagWithNewlines"],
        messages,
        false
    );
    
    // Should handle special characters without crashing
    EXPECT_FALSE(result.empty());
    EXPECT_NE(result.find("quotes"), std::string::npos);
}

TEST_F(ChatTemplateIntegrationTest, GenerationPromptToggle) {
    std::vector<ChatMessage> messages = {
        {"user", "Question"},
        {"assistant", "Answer"}
    };
    
    std::string with_prompt = tok.apply_chat_template(
        templates["HeadersWithContent"],
        messages,
        true
    );
    
    std::string without_prompt = tok.apply_chat_template(
        templates["HeadersWithContent"],
        messages,
        false
    );
    
    // Both should contain message content
    EXPECT_NE(with_prompt.find("Question"), std::string::npos);
    EXPECT_NE(without_prompt.find("Question"), std::string::npos);
    
    // They may differ in length/format (with_prompt may have extra markers)
    // Just verify they both work
}
