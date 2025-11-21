#pragma once
#include <string>

// Common utilities for all tests
namespace test_utils {
    std::string find_resource(const std::string& name);
}

// Test function signatures - return 0 on success, non-zero on failure
int test_basic();
int test_vocab_size();
int test_special_token_encode();
int test_encode_variations();
int test_error_handling();
int test_bert_tokenizer();
int test_serialization_decoding_batch();
