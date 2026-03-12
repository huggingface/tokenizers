#include "tokenizers/tokenizers.h"
#include <iostream>
#include <vector>
#include <string>

using namespace tokenizers;

int main(int argc, char* argv[]) {
    // Check if tokenizer path is provided
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_tokenizer.json> [text_to_encode]\n";
        std::cerr << "\nExample:\n";
        std::cerr << "  " << argv[0] << " ../../tokenizers/data/tokenizer.json \"Hello world!\"\n";
        return 1;
    }

    std::string tokenizer_path = argv[1];
    std::string text = (argc >= 3) ? argv[2] : "Hello, world!";

    // Print version information
    std::cout << "Tokenizers C++ Bindings Version: " << Tokenizer::version() << "\n\n";

    // Load the tokenizer
    std::cout << "Loading tokenizer from: " << tokenizer_path << "\n";
    Tokenizer tokenizer(tokenizer_path);
    
    if (!tokenizer.valid()) {
        std::cerr << "Error: Failed to load tokenizer from " << tokenizer_path << "\n";
        std::cerr << "Make sure the file exists and is a valid tokenizer JSON file.\n";
        return 1;
    }

    std::cout << "✓ Tokenizer loaded successfully\n\n";

    // Get vocabulary size
    size_t vocab_size = tokenizer.vocab_size();
    std::cout << "Vocabulary size: " << vocab_size << "\n\n";

    // Example 1: Basic encoding
    std::cout << "=== Example 1: Basic Encoding ===\n";
    std::cout << "Input text: \"" << text << "\"\n";
    
    auto ids_with_special = tokenizer.encode(text, true);
    std::cout << "Tokens (with special tokens): [";
    for (size_t i = 0; i < ids_with_special.size(); ++i) {
        std::cout << ids_with_special[i];
        if (i + 1 < ids_with_special.size()) std::cout << ", ";
    }
    std::cout << "]\n";
    std::cout << "Token count: " << ids_with_special.size() << "\n\n";

    // Example 2: Encoding without special tokens
    std::cout << "=== Example 2: Encoding Without Special Tokens ===\n";
    auto ids_without_special = tokenizer.encode(text, false);
    std::cout << "Tokens (without special tokens): [";
    for (size_t i = 0; i < ids_without_special.size(); ++i) {
        std::cout << ids_without_special[i];
        if (i + 1 < ids_without_special.size()) std::cout << ", ";
    }
    std::cout << "]\n";
    std::cout << "Token count: " << ids_without_special.size() << "\n\n";

    // Example 3: Token lookup
    std::cout << "=== Example 3: Token ID Lookup ===\n";
    std::vector<std::string> sample_tokens = {"hello", "world", "the", "[UNK]", "[PAD]"};
    for (const auto& token : sample_tokens) {
        int32_t id = tokenizer.token_to_id(token);
        if (id >= 0) {
            std::cout << "Token \"" << token << "\" -> ID: " << id << "\n";
        } else {
            std::cout << "Token \"" << token << "\" -> Not found in vocabulary\n";
        }
    }
    std::cout << "\n";

    // Example 4: Adding special tokens
    std::cout << "=== Example 4: Adding Custom Special Token ===\n";
    std::string new_token = "[CUSTOM_TOKEN]";
    size_t vocab_before = tokenizer.vocab_size();
    bool added = tokenizer.add_special_token(new_token);
    size_t vocab_after = tokenizer.vocab_size();
    
    if (added) {
        std::cout << "✓ Successfully added special token: " << new_token << "\n";
        std::cout << "Vocabulary size increased: " << vocab_before << " -> " << vocab_after << "\n";
        
        int32_t new_id = tokenizer.token_to_id(new_token);
        std::cout << "New token ID: " << new_id << "\n\n";
        
        // Encode text with the new token
        std::string text_with_token = "Hello " + new_token + " world";
        auto ids = tokenizer.encode(text_with_token, true);
        std::cout << "Encoding \"" << text_with_token << "\":\n";
        std::cout << "Token IDs: [";
        for (size_t i = 0; i < ids.size(); ++i) {
            std::cout << ids[i];
            if (i + 1 < ids.size()) std::cout << ", ";
        }
        std::cout << "]\n";
    } else {
        std::cout << "Failed to add special token (may already exist)\n";
    }
    std::cout << "\n";

    // Example 5: Batch encoding multiple texts
    std::cout << "=== Example 5: Encoding Multiple Texts ===\n";
    std::vector<std::string> texts = {
        "The quick brown fox",
        "jumps over the lazy dog",
        "Hello, world!",
        "Testing tokenization"
    };
    
    for (const auto& t : texts) {
        auto tokens = tokenizer.encode(t, true);
        std::cout << "\"" << t << "\" -> " << tokens.size() << " tokens\n";
    }
    std::cout << "\n";

    // Example 6: Move semantics
    std::cout << "=== Example 6: Move Semantics ===\n";
    Tokenizer moved_tokenizer = std::move(tokenizer);
    std::cout << "Original tokenizer valid: " << (tokenizer.valid() ? "yes" : "no") << "\n";
    std::cout << "Moved tokenizer valid: " << (moved_tokenizer.valid() ? "yes" : "no") << "\n";
    std::cout << "Moved tokenizer vocab size: " << moved_tokenizer.vocab_size() << "\n\n";

    std::cout << "=== All Examples Completed Successfully ===\n";
    return 0;
}
