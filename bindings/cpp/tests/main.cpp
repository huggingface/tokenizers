#include "test_common.h"
#include <iostream>
#include <string>
#include <map>

// Test registry
static const std::map<std::string, int(*)()> test_registry = {
    {"basic", test_basic},
    {"vocab_size", test_vocab_size},
    {"special_token_encode", test_special_token_encode},
    {"encode_variations", test_encode_variations},
    {"error_handling", test_error_handling},
    {"bert_tokenizer", test_bert_tokenizer},
    {"serialization_decoding_batch", test_serialization_decoding_batch},
};

void print_usage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name << " <test_name>\n";
    std::cerr << "Available tests:\n";
    for (const auto& entry : test_registry) {
        std::cerr << "  - " << entry.first << "\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string test_name = argv[1];
    auto it = test_registry.find(test_name);
    if (it == test_registry.end()) {
        std::cerr << "Unknown test: " << test_name << "\n";
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "Running test: " << test_name << "\n";
    int result = it->second();
    if (result == 0) {
        std::cout << "✓ Test " << test_name << " passed\n";
    } else {
        std::cerr << "✗ Test " << test_name << " failed with code " << result << "\n";
    }
    return result;
}
