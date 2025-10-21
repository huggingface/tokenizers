#include "test_common.h"
#include "tokenizers/tokenizers.h"
#include <cassert>
#include <iostream>
#include <vector>

using namespace tokenizers;
using test_utils::find_resource;

int test_bert_tokenizer() {
    auto path = find_resource("bert-wiki.json");
    assert(!path.empty() && "Resource bert-wiki.json not found; run make -C tokenizers test");
    
    Tokenizer tok(path);
    assert(tok.valid());
    
    size_t v1 = tok.vocab_size();
    std::cout << "Initial vocab size: " << v1 << "\n";
    assert(v1 > 0 && "Vocab size should be positive");
    
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
        assert(!ids.empty() && "Each encoding should produce tokens");
        std::cout << "\"" << text << "\" -> " << ids.size() << " tokens\n";
    }
    
    // Test that adding duplicate special token doesn't break things
    tok.add_special_token("[SPECIAL1]");
    tok.add_special_token("[SPECIAL1]"); // duplicate
    tok.add_special_token("[SPECIAL2]");
    
    int32_t id1a = tok.token_to_id("[SPECIAL1]");
    int32_t id1b = tok.token_to_id("[SPECIAL1]");
    int32_t id2 = tok.token_to_id("[SPECIAL2]");
    
    assert(id1a == id1b && "Same token should have same id");
    assert(id1a >= 0 && id2 >= 0 && "Special tokens should have valid ids");
    assert(id1a != id2 && "Different tokens should have different ids");
    
    std::cout << "BERT tokenizer integration test passed.\n";
    return 0;
}
