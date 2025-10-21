#include "test_common.h"
#include "tokenizers/tokenizers.h"
#include <cassert>
#include <iostream>

using namespace tokenizers;
using test_utils::find_resource;

int test_basic() {
    std::cout << "Version: " << Tokenizer::version() << "\n";

    // Use tokenizer.json which exists after running `make -C tokenizers test`
    auto path = find_resource("tokenizer.json");
    assert(!path.empty() && "Failed to locate tokenizer resource tokenizer.json. Run `make -C tokenizers test` first.");

    Tokenizer tok(path);
    assert(tok.valid() && "Failed to load tokenizer JSON file");

    auto ids = tok.encode("Hello world!");
    assert(!ids.empty() && "Encoding produced no ids");

    // Basic sanity: ids should be positive.
    bool any_non_negative = false;
    for (auto id : ids) {
        if (id >= 0) { any_non_negative = true; break; }
    }
    assert(any_non_negative && "No non-negative token ids found, unexpected");

    std::cout << "Encoded Hello world! -> [";
    for (size_t i = 0; i < ids.size(); ++i) {
        std::cout << ids[i];
        if (i + 1 < ids.size()) std::cout << ", ";
    }
    std::cout << "]\nTest passed.\n";
    return 0;
}
