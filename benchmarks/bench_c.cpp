#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <string>
#include <cstdlib>

// Include the C FFI header
extern "C" {
    #include "../bindings/c/tokenizers_c.h"
}

std::string read_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <tokenizer.json> <input.txt>" << std::endl;
        return 1;
    }
    
    std::string tokenizer_path = argv[1];
    std::string input_path = argv[2];
    
    try {
        // Load tokenizer
        auto load_start = std::chrono::high_resolution_clock::now();
        void* tokenizer = tokenizers_new_from_file(tokenizer_path.c_str());
        if (!tokenizer) {
            throw std::runtime_error("Failed to load tokenizer from file: " + tokenizer_path);
        }
        auto load_end = std::chrono::high_resolution_clock::now();
        auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start);
        
        // Read input file
        std::string text = read_file(input_path);
        
        // Benchmark encoding
        auto encode_start = std::chrono::high_resolution_clock::now();
        tokenizers_encoding_t encoding = tokenizers_encode(tokenizer, text.c_str(), false);
        auto encode_end = std::chrono::high_resolution_clock::now();
        auto encode_time = std::chrono::duration_cast<std::chrono::milliseconds>(encode_end - encode_start);
        
        if (!encoding.ids) {
            tokenizers_free(tokenizer);
            throw std::runtime_error("Failed to encode text");
        }
        
        size_t num_tokens = encoding.len;
        size_t num_chars = text.length();
        double tokens_per_sec = num_tokens / (encode_time.count() / 1000.0);
        
        // Print results in a parseable format
        std::cout << "load_time_ms:" << load_time.count() << std::endl;
        std::cout << "encode_time_ms:" << encode_time.count() << std::endl;
        std::cout << "num_tokens:" << num_tokens << std::endl;
        std::cout << "num_chars:" << num_chars << std::endl;
        std::cout << "tokens_per_sec:" << std::fixed << tokens_per_sec << std::endl;
        
        // Cleanup
        tokenizers_free_encoding(encoding);
        tokenizers_free(tokenizer);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
