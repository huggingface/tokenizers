/**
 * Tokenizer C++ bindings tests
 */
#include <gtest/gtest.h>
#include <tokenizers/tokenizers.h>
#include "test_common.h"
#include <filesystem>

using namespace tokenizers;
using test_utils::find_resource;

// ==================== Basic Tokenizer Tests ====================

class TokenizerTest : public ::testing::Test {
protected:
    Tokenizer tok;
    
    void SetUp() override {
        std::string path = find_resource("tokenizer.json");
        ASSERT_FALSE(path.empty()) << "Could not find tokenizer.json";
        tok = Tokenizer(path);
        ASSERT_TRUE(tok.valid());
    }
};

TEST_F(TokenizerTest, Encode) {
    auto output = tok.encode("my name is john");
    EXPECT_FALSE(output.ids.empty());
    EXPECT_EQ(output.ids.size(), output.attention_mask.size());
    
    // Consistency check - same input gives same output
    EXPECT_EQ(tok.encode("my name is john"), output);
}

TEST_F(TokenizerTest, EncodeBatch) {
    std::vector<std::string> batch = {"my name is john", "my pair"};
    auto output = tok.encode_batch(batch);
    ASSERT_EQ(output.size(), 2);
    EXPECT_FALSE(output[0].ids.empty());
    EXPECT_FALSE(output[1].ids.empty());
}

TEST_F(TokenizerTest, Decode) {
    auto encoding = tok.encode("my name is john");
    auto decoded = tok.decode(encoding.ids);
    EXPECT_NE(decoded.find("name"), std::string::npos);
    EXPECT_NE(decoded.find("john"), std::string::npos);
}

TEST_F(TokenizerTest, DecodeBatch) {
    std::vector<std::string> batch = {"my name is john", "my pair"};
    auto encodings = tok.encode_batch(batch);
    
    std::vector<std::vector<int32_t>> batch_ids;
    for (const auto& enc : encodings) batch_ids.push_back(enc.ids);
    
    auto decoded = tok.decode_batch(batch_ids);
    ASSERT_EQ(decoded.size(), 2);
    EXPECT_NE(decoded[0].find("john"), std::string::npos);
    EXPECT_NE(decoded[1].find("pair"), std::string::npos);
}

TEST_F(TokenizerTest, Vocab) {
    EXPECT_GT(tok.vocab_size(), 0);
    
    int32_t id = tok.token_to_id("the");
    if (id != -1) {
        EXPECT_EQ(tok.id_to_token(id), "the");
    }
}

TEST_F(TokenizerTest, Padding) {
    PaddingParams params;
    params.strategy = PaddingParams::Fixed;
    params.fixed_length = 10;
    params.pad_id = 0;
    tok.set_padding(params);
    
    auto output = tok.encode("short");
    EXPECT_EQ(output.ids.size(), 10);
    
    tok.disable_padding();
    EXPECT_LT(tok.encode("short").ids.size(), 10);
}

TEST_F(TokenizerTest, AddSpecialTokens) {
    size_t added = tok.add_special_tokens({"[SPECIAL1]", "[SPECIAL2]"});
    EXPECT_EQ(added, 2);
    
    int32_t id = tok.token_to_id("[SPECIAL1]");
    EXPECT_NE(id, -1);
    
    auto output = tok.encode("Hello [SPECIAL1]");
    EXPECT_NE(std::find(output.ids.begin(), output.ids.end(), id), output.ids.end());
}

TEST_F(TokenizerTest, SaveAndLoad) {
    std::string save_path = "test_save_gtest.json";
    EXPECT_TRUE(tok.save(save_path));
    
    Tokenizer t2(save_path);
    EXPECT_TRUE(t2.valid());
    EXPECT_EQ(t2.vocab_size(), tok.vocab_size());
    
    std::filesystem::remove(save_path);
}

TEST_F(TokenizerTest, ToStringAndFromBlob) {
    std::string json = tok.to_string(false);
    EXPECT_FALSE(json.empty());
    
    Tokenizer t2 = Tokenizer::FromBlobJSON(json);
    EXPECT_TRUE(t2.valid());
    EXPECT_EQ(t2.vocab_size(), tok.vocab_size());
}

TEST_F(TokenizerTest, SpecialTokensFromConfig) {
    // Config should be auto-loaded from tokenizer_config.json
    EXPECT_EQ(tok.bos_token(), "<bos>");
    EXPECT_EQ(tok.eos_token(), "<eos>");
    EXPECT_EQ(tok.pad_token(), "<pad>");
    EXPECT_EQ(tok.unk_token(), "<unk>");
    
    EXPECT_GE(tok.bos_id(), 0);
    EXPECT_GE(tok.eos_id(), 0);
    EXPECT_GE(tok.pad_id(), 0);
    EXPECT_GE(tok.unk_id(), 0);
    
    EXPECT_TRUE(tok.add_bos_token());
    EXPECT_FALSE(tok.add_eos_token());
}

TEST_F(TokenizerTest, ChatTemplate) {
    EXPECT_TRUE(tok.has_chat_template());
    EXPECT_FALSE(tok.chat_template().empty());
    
    std::vector<ChatMessage> messages = {
        {"user", "Hello!"},
        {"assistant", "Hi there!"},
        {"user", "How are you?"}
    };
    
    std::string result = tok.apply_chat_template(messages, true);
    EXPECT_NE(result.find("Hello!"), std::string::npos);
    EXPECT_NE(result.find("Hi there!"), std::string::npos);
    EXPECT_NE(result.find("How are you?"), std::string::npos);
}

// ==================== BERT Tokenizer Tests ====================

class BertTokenizerTest : public ::testing::Test {
protected:
    Tokenizer tok;
    
    void SetUp() override {
        std::string path = find_resource("bert-wiki.json");
        ASSERT_FALSE(path.empty()) << "Could not find bert-wiki.json";
        // Pass empty config path to skip loading tokenizer_config.json
        tok = Tokenizer(path, "");
        ASSERT_TRUE(tok.valid());
    }
};

TEST_F(BertTokenizerTest, SpecialTokensViaHeuristic) {
    // BERT tokens found via heuristic (no config file)
    EXPECT_EQ(tok.id_to_token(tok.bos_id()), "[CLS]");
    EXPECT_EQ(tok.id_to_token(tok.eos_id()), "[SEP]");
    EXPECT_EQ(tok.id_to_token(tok.pad_id()), "[PAD]");
    EXPECT_EQ(tok.id_to_token(tok.unk_id()), "[UNK]");
    
    // IDs should match token_to_id
    EXPECT_EQ(tok.bos_id(), tok.token_to_id("[CLS]"));
    EXPECT_EQ(tok.eos_id(), tok.token_to_id("[SEP]"));
    EXPECT_EQ(tok.pad_id(), tok.token_to_id("[PAD]"));
    EXPECT_EQ(tok.unk_id(), tok.token_to_id("[UNK]"));
}

TEST_F(BertTokenizerTest, ExplicitConfigPath) {
    auto config_path = find_resource("bert_tokenizer_config.json");
    if (config_path.empty()) {
        GTEST_SKIP() << "bert_tokenizer_config.json not found";
    }
    
    auto tok_path = find_resource("bert-wiki.json");
    Tokenizer tok_with_config(tok_path, config_path);
    ASSERT_TRUE(tok_with_config.valid());
    
    EXPECT_EQ(tok_with_config.bos_token(), "[CLS]");
    EXPECT_EQ(tok_with_config.eos_token(), "[SEP]");
    EXPECT_FALSE(tok_with_config.has_chat_template());
}

TEST_F(BertTokenizerTest, NoChatTemplate) {
    EXPECT_FALSE(tok.has_chat_template());
    
    std::vector<ChatMessage> messages = {{"user", "Hello!"}};
    EXPECT_THROW(tok.apply_chat_template(messages), ChatTemplateError);
}

// ==================== Error Handling Tests ====================

TEST(TokenizerErrorTest, InvalidFile) {
    Tokenizer tok("nonexistent_file.json");
    EXPECT_FALSE(tok.valid());
    
    // All operations should return safe defaults
    EXPECT_EQ(tok.vocab_size(), 0);
    EXPECT_TRUE(tok.encode("test").empty());
    EXPECT_EQ(tok.token_to_id("test"), -1);
    EXPECT_EQ(tok.bos_id(), -1);
    EXPECT_TRUE(tok.bos_token().empty());
    EXPECT_FALSE(tok.has_chat_template());
}

TEST(TokenizerErrorTest, MoveSemantics) {
    auto path = find_resource("tokenizer.json");
    ASSERT_FALSE(path.empty());
    
    Tokenizer tok(path);
    EXPECT_TRUE(tok.valid());
    
    Tokenizer moved = std::move(tok);
    EXPECT_TRUE(moved.valid());
    EXPECT_FALSE(tok.valid());
}

TEST(TokenizerErrorTest, UnknownToken) {
    auto path = find_resource("tokenizer.json");
    ASSERT_FALSE(path.empty());
    
    Tokenizer tok(path);
    EXPECT_EQ(tok.token_to_id("[DEFINITELY_NOT_IN_VOCAB_12345]"), -1);
}

TEST(TokenizerErrorTest, FromBlobNoChatTemplate) {
    // Tokenizer loaded from string has no config
    std::string json = R"({
        "version": "1.0",
        "added_tokens": [{"id": 0, "content": "[UNK]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}],
        "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "hello": 1}, "unk_token": "[UNK]"}
    })";
    
    Tokenizer tok = Tokenizer::FromBlobJSON(json);
    ASSERT_TRUE(tok.valid());
    EXPECT_FALSE(tok.has_chat_template());
}

// ==================== Optional Tokenizer Tests ====================

TEST(OptionalTokenizerTest, Llama) {
    auto path = find_resource("llama-3-tokenizer.json");
    if (path.empty()) {
        GTEST_SKIP() << "llama-3-tokenizer.json not found";
    }
    
    Tokenizer tok(path);
    ASSERT_TRUE(tok.valid());
    
    int32_t bos = tok.bos_id();
    if (bos >= 0) {
        std::string bos_token = tok.id_to_token(bos);
        EXPECT_TRUE(bos_token == "<|begin_of_text|>" || bos_token == "<s>");
    }
}

TEST(OptionalTokenizerTest, Unigram) {
    auto path = find_resource("unigram.json");
    if (path.empty()) {
        GTEST_SKIP() << "unigram.json not found";
    }
    
    Tokenizer tok(path);
    if (!tok.valid()) {
        GTEST_SKIP() << "unigram.json is not a complete tokenizer file";
    }
    
    // Just verify API doesn't crash
    tok.bos_id();
    tok.eos_id();
    tok.unk_id();
}

