#include <gtest/gtest.h>
#include "tokenizers/tokenizers.h"
#include "test_common.h"
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>

using namespace tokenizers;
using test_utils::find_resource;

class TokenizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::string path = find_resource("tokenizer.json");
        ASSERT_FALSE(path.empty()) << "Could not find tokenizer.json";
        tokenizer = std::make_unique<Tokenizer>(path);
        ASSERT_TRUE(tokenizer->valid());
    }

    std::unique_ptr<Tokenizer> tokenizer;
};

TEST_F(TokenizerTest, TestEncode) {
    // Can encode single sequence
    auto output = tokenizer->encode("my name is john");
    EXPECT_FALSE(output.ids.empty());
    EXPECT_FALSE(output.attention_mask.empty());
    EXPECT_EQ(output.ids.size(), output.attention_mask.size());

    // Verify specific tokens if possible, but ids depend on the model
    // For "tokenizer.json" (roberta-base), "my" -> 127, "name" -> 766, "is" -> 16, "john" -> 619
    // Note: The tokenizer.json in data might be different.
    // Let's just check structure for now.
}

TEST_F(TokenizerTest, TestEncodeBatch) {
    std::vector<std::string> batch = {"my name is john", "my pair"};
    auto output = tokenizer->encode_batch(batch);
    ASSERT_EQ(output.size(), 2);
    EXPECT_FALSE(output[0].ids.empty());
    EXPECT_FALSE(output[1].ids.empty());
}

TEST_F(TokenizerTest, TestDecode) {
    auto encoding = tokenizer->encode("my name is john");
    auto decoded = tokenizer->decode(encoding.ids);
    // The tokenizer.json is likely a BPE/RoBERTa, so it might preserve spaces or add prefixes
    // We check if the decoded string contains the original words
    EXPECT_NE(decoded.find("name"), std::string::npos);
    EXPECT_NE(decoded.find("john"), std::string::npos);
}

TEST_F(TokenizerTest, TestDecodeBatch) {
    std::vector<std::string> batch = {"my name is john", "my pair"};
    auto encodings = tokenizer->encode_batch(batch);
    
    std::vector<std::vector<int32_t>> batch_ids;
    for (const auto& enc : encodings) batch_ids.push_back(enc.ids);
    
    auto decoded = tokenizer->decode_batch(batch_ids);
    ASSERT_EQ(decoded.size(), 2);
    EXPECT_NE(decoded[0].find("john"), std::string::npos);
    EXPECT_NE(decoded[1].find("pair"), std::string::npos);
}

TEST_F(TokenizerTest, TestVocab) {
    size_t size = tokenizer->vocab_size();
    EXPECT_GT(size, 0);

    int32_t id = tokenizer->token_to_id("the");
    // "the" is usually in vocab
    if (id != -1) {
        std::string token = tokenizer->id_to_token(id);
        EXPECT_EQ(token, "the");
    }
}

TEST_F(TokenizerTest, TestPadding) {
    PaddingParams params;
    params.strategy = PaddingParams::Fixed;
    params.fixed_length = 10;
    params.pad_id = 0;
    
    tokenizer->set_padding(params);
    
    auto output = tokenizer->encode("short");
    EXPECT_EQ(output.ids.size(), 10);
    EXPECT_EQ(output.attention_mask.size(), 10);
    
    // Check padding
    int padding_count = 0;
    for (auto mask : output.attention_mask) {
        if (mask == 0) padding_count++;
    }
    EXPECT_GT(padding_count, 0);
    
    tokenizer->disable_padding();
    auto output_no_pad = tokenizer->encode("short");
    EXPECT_LT(output_no_pad.ids.size(), 10);
}

TEST_F(TokenizerTest, TestAddSpecialTokens) {
    std::vector<std::string> specials = {"[SPECIAL1]", "[SPECIAL2]"};
    size_t added = tokenizer->add_special_tokens(specials);
    EXPECT_EQ(added, 2);
    
    int32_t id1 = tokenizer->token_to_id("[SPECIAL1]");
    EXPECT_NE(id1, -1);
    
    auto output = tokenizer->encode("Hello [SPECIAL1]");
    bool found = false;
    for (auto id : output.ids) {
        if (id == id1) found = true;
    }
    EXPECT_TRUE(found);
}

TEST_F(TokenizerTest, TestSave) {
    std::string save_path = "test_save_gtest.json";
    EXPECT_TRUE(tokenizer->save(save_path));
    
    Tokenizer t2(save_path);
    EXPECT_TRUE(t2.valid());
    EXPECT_EQ(t2.vocab_size(), tokenizer->vocab_size());
    
    std::filesystem::remove(save_path);
}

TEST_F(TokenizerTest, TestToString) {
    std::string json = tokenizer->to_string(false);
    EXPECT_FALSE(json.empty());
    EXPECT_NE(json.find("version"), std::string::npos);
    
    Tokenizer t2 = Tokenizer::FromBlobJSON(json);
    EXPECT_TRUE(t2.valid());
}

TEST_F(TokenizerTest, TestVocabSizeGrowth) {
    size_t v1 = tokenizer->vocab_size();
    // Add a special token and expect vocab size to grow by at least 1.
    bool added = tokenizer->add_special_token("[NEW_SPECIAL]");
    EXPECT_TRUE(added);
    size_t v2 = tokenizer->vocab_size();
    EXPECT_GE(v2, v1 + 1);

    int32_t id = tokenizer->token_to_id("[NEW_SPECIAL]");
    EXPECT_GE(id, 0);
}

TEST_F(TokenizerTest, TestSpecialTokenEncode) {
    // Add special token and then encode a string containing it.
    const std::string special = "[FOO_BAR]";
    bool ok = tokenizer->add_special_token(special);
    EXPECT_TRUE(ok);
    int32_t special_id = tokenizer->token_to_id(special);
    EXPECT_GE(special_id, 0);

    std::string input = "Hello " + special + " world";
    auto ids = tokenizer->encode(input);
    EXPECT_FALSE(ids.empty());
    bool present = std::find(ids.begin(), ids.end(), special_id) != ids.end();
    EXPECT_TRUE(present);
}

TEST_F(TokenizerTest, TestEncodeVariations) {
    // Test encode with and without special tokens
    std::string text = "Hello world!";
    auto ids_with = tokenizer->encode(text, true);
    auto ids_without = tokenizer->encode(text, false);
    
    EXPECT_FALSE(ids_with.empty());
    EXPECT_FALSE(ids_without.empty());
    
    // Test empty input
    auto empty_ids = tokenizer->encode("", true);
    // Empty input may still produce special tokens depending on tokenizer config
    
    // Test repeated encoding (consistency check)
    auto ids_again = tokenizer->encode(text, true);
    EXPECT_EQ(ids_again, ids_with);
}

TEST_F(TokenizerTest, TestErrorHandling) {
    // Test invalid file loading
    Tokenizer bad_tok("nonexistent_file.json");
    EXPECT_FALSE(bad_tok.valid());
    
    // Verify operations on invalid tokenizer return safe defaults
    EXPECT_EQ(bad_tok.vocab_size(), 0);
    EXPECT_TRUE(bad_tok.encode("test").empty());
    EXPECT_EQ(bad_tok.token_to_id("test"), -1);
    
    // Look up a token that definitely doesn't exist in vocab
    std::string fake_token = "[DEFINITELY_NOT_IN_VOCAB_12345]";
    int32_t id = tokenizer->token_to_id(fake_token);
    EXPECT_EQ(id, -1);
    
    // Test move semantics
    Tokenizer moved = std::move(*tokenizer);
    EXPECT_TRUE(moved.valid());
    // Original tokenizer should be invalid after move (or at least handle_ is null)
    // But since we moved from a unique_ptr managed object, we need to be careful.
    // The test logic in test_error_handling.cpp moved a stack object.
    // Here tokenizer is a unique_ptr.
    // Let's create a local tokenizer for this test.
    
    std::string path = find_resource("tokenizer.json");
    Tokenizer tok(path);
    EXPECT_TRUE(tok.valid());
    Tokenizer moved_tok = std::move(tok);
    EXPECT_TRUE(moved_tok.valid());
    EXPECT_FALSE(tok.valid());
}

TEST_F(TokenizerTest, TestBertTokenizer) {
    auto path = find_resource("bert-wiki.json");
    ASSERT_FALSE(path.empty());
    
    Tokenizer tok(path);
    ASSERT_TRUE(tok.valid());
    
    size_t v1 = tok.vocab_size();
    EXPECT_GT(v1, 0);
    
    // Test multiple encodings with different texts
    std::vector<std::string> test_cases = {
        "The quick brown fox",
        "jumps over the lazy dog",
        "Hello, world!",
        "Testing tokenization with punctuation: !@#$%",
        "Numbers: 123 456 789"
    };
    
    for (const auto& text : test_cases) {
        auto ids = tok.encode(text, true);
        EXPECT_FALSE(ids.empty());
    }
    
    // Test that adding duplicate special token doesn't break things
    tok.add_special_token("[SPECIAL1]");
    tok.add_special_token("[SPECIAL1]"); // duplicate
    tok.add_special_token("[SPECIAL2]");
    
    int32_t id1a = tok.token_to_id("[SPECIAL1]");
    int32_t id1b = tok.token_to_id("[SPECIAL1]");
    int32_t id2 = tok.token_to_id("[SPECIAL2]");
    
    EXPECT_EQ(id1a, id1b);
    EXPECT_GE(id1a, 0);
    EXPECT_GE(id2, 0);
    EXPECT_NE(id1a, id2);
}

