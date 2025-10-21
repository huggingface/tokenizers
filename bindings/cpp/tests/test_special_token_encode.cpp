#include "test_common.h"
#include "tokenizers/tokenizers.h"
#include <cassert>
#include <iostream>
#include <algorithm>

using namespace tokenizers;
using test_utils::find_resource;

int test_special_token_encode() {
    auto path = find_resource("tokenizer.json");
    assert(!path.empty() && "Resource tokenizer.json not found; run make -C tokenizers test");
    Tokenizer tok(path);
    assert(tok.valid());

    // Add special token and then encode a string containing it.
    const std::string special = "[FOO_BAR]";
    bool ok = tok.add_special_token(special);
    assert(ok && "Failed to add special token");
    int32_t special_id = tok.token_to_id(special);
    assert(special_id >= 0 && "Special token should have a valid id");

    std::string input = "Hello " + special + " world";
    auto ids = tok.encode(input);
    assert(!ids.empty());
    bool present = std::find(ids.begin(), ids.end(), special_id) != ids.end();
    assert(present && "Encoded ids should contain the special token id when token appears in input");

    std::cout << "Special token id: " << special_id << " present in encoding (size=" << ids.size() << ")\n";
    return 0;
}
