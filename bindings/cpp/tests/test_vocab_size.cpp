#include "test_common.h"
#include "tokenizers/tokenizers.h"
#include <cassert>
#include <iostream>

using namespace tokenizers;
using test_utils::find_resource;

int test_vocab_size() {
    auto path = find_resource("tokenizer.json");
    assert(!path.empty() && "Resource tokenizer.json not found; run make -C tokenizers test");
    Tokenizer tok(path);
    assert(tok.valid());

    size_t v1 = tok.vocab_size();
    // Add a special token and expect vocab size to grow by at least 1.
    bool added = tok.add_special_token("[NEW_SPECIAL]");
    assert(added && "Failed to add special token");
    size_t v2 = tok.vocab_size();
    assert(v2 >= v1 + 1 && "Vocab size did not increase after adding special token");

    int32_t id = tok.token_to_id("[NEW_SPECIAL]");
    assert(id >= 0 && "Token ID for newly added special token should be non-negative");

    std::cout << "Initial vocab: " << v1 << ", after add: " << v2 << ", new token id: " << id << "\n";
    return 0;
}
