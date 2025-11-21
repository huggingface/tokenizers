#include "test_common.h"
#include "tokenizers/tokenizers.h"
#include <cassert>
#include <iostream>
#include <fstream>

using namespace tokenizers;
using test_utils::find_resource;

int test_serialization_decoding_batch() {
    auto path = find_resource("tokenizer.json");
    assert(!path.empty());

    Tokenizer tok(path);
    assert(tok.valid());

    // Test id_to_token
    auto ids = tok.encode("Hello");
    assert(!ids.empty());
    int32_t id = ids[0];
    std::string token = tok.id_to_token(id);
    assert(!token.empty());
    std::cout << "id_to_token(" << id << ") = " << token << "\n";

    // Test decode
    std::string decoded = tok.decode(ids);
    std::cout << "decode(" << ids.size() << " ids) = " << decoded << "\n";
    assert(!decoded.empty());

    // Test to_string
    std::string json = tok.to_string(false);
    assert(!json.empty());
    assert(json.find("version") != std::string::npos);

    // Test FromBlobJSON
    Tokenizer tok2 = Tokenizer::FromBlobJSON(json);
    assert(tok2.valid());
    assert(tok2.vocab_size() == tok.vocab_size());

    // Test save
    std::string save_path = "test_save_tokenizer.json";
    bool saved = tok.save(save_path, true);
    assert(saved);
    
    Tokenizer tok3(save_path);
    assert(tok3.valid());
    assert(tok3.vocab_size() == tok.vocab_size());

    // Test add_special_tokens
    std::vector<std::string> new_special_tokens = {"[SPECIAL1]", "[SPECIAL2]"};
    size_t added = tok.add_special_tokens(new_special_tokens);
    assert(added == 2);
    assert(tok.token_to_id("[SPECIAL1]") != -1);
    assert(tok.token_to_id("[SPECIAL2]") != -1);

    // Test encode_batch
    std::vector<std::string> batch_texts = {"Hello world", "Hello [SPECIAL1]"};
    auto batch_ids = tok.encode_batch(batch_texts);
    assert(batch_ids.size() == 2);
    assert(!batch_ids[0].empty());
    assert(!batch_ids[1].empty());
    // Check if [SPECIAL1] is encoded correctly
    int32_t special_id = tok.token_to_id("[SPECIAL1]");
    bool found_special = false;
    for (auto id : batch_ids[1]) {
        if (id == special_id) {
            found_special = true;
            break;
        }
    }
    assert(found_special);

    std::cout << "New features test passed.\n";
    return 0;
}
