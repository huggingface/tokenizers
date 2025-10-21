#include "test_common.h"
#include "tokenizers/tokenizers.h"
#include <cassert>
#include <iostream>

using namespace tokenizers;
using test_utils::find_resource;

int test_encode_variations() {
    auto path = find_resource("tokenizer.json");
    assert(!path.empty() && "Resource tokenizer.json not found; run make -C tokenizers test");
    Tokenizer tok(path);
    assert(tok.valid());

    // Test encode with and without special tokens
    std::string text = "Hello world!";
    auto ids_with = tok.encode(text, true);
    auto ids_without = tok.encode(text, false);
    
    assert(!ids_with.empty());
    assert(!ids_without.empty());
    
    // Usually encoding with special tokens adds more tokens
    std::cout << "With special tokens: " << ids_with.size() << " ids\n";
    std::cout << "Without special tokens: " << ids_without.size() << " ids\n";
    
    // Test empty input
    auto empty_ids = tok.encode("", true);
    // Empty input may still produce special tokens depending on tokenizer config
    std::cout << "Empty input produced: " << empty_ids.size() << " ids\n";
    
    // Test repeated encoding (consistency check)
    auto ids_again = tok.encode(text, true);
    assert(ids_again == ids_with && "Repeated encoding should produce identical results");
    
    std::cout << "Encode variations test passed.\n";
    return 0;
}
