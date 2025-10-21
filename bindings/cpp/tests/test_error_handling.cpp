#include "test_common.h"
#include "tokenizers/tokenizers.h"
#include <cassert>
#include <iostream>

using namespace tokenizers;
using test_utils::find_resource;

int test_error_handling() {
    // Test invalid file loading
    Tokenizer bad_tok("nonexistent_file.json");
    assert(!bad_tok.valid() && "Should fail to load nonexistent file");
    
    // Verify operations on invalid tokenizer return safe defaults
    assert(bad_tok.vocab_size() == 0 && "Invalid tokenizer should return 0 vocab size");
    assert(bad_tok.encode("test").empty() && "Invalid tokenizer should return empty encoding");
    assert(bad_tok.token_to_id("test") == -1 && "Invalid tokenizer should return -1 for token_to_id");
    
    // Test valid tokenizer with nonexistent token
    auto path = find_resource("tokenizer.json");
    assert(!path.empty() && "Resource tokenizer.json not found; run make -C tokenizers test");
    Tokenizer tok(path);
    assert(tok.valid());
    
    // Look up a token that definitely doesn't exist in vocab
    std::string fake_token = "[DEFINITELY_NOT_IN_VOCAB_12345]";
    int32_t id = tok.token_to_id(fake_token);
    assert(id == -1 && "Nonexistent token should return -1");
    
    // Test move semantics
    Tokenizer moved = std::move(tok);
    assert(moved.valid() && "Moved tokenizer should be valid");
    assert(!tok.valid() && "Original tokenizer should be invalid after move");
    
    std::cout << "Error handling test passed.\n";
    return 0;
}
