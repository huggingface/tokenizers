#ifndef TOKENIZERS_C_H
#define TOKENIZERS_C_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const int* ids;
    size_t len;
} tokenizers_encoding_t;

// Create a new tokenizer from a JSON file
void* tokenizers_new_from_file(const char* path);

// Create a new tokenizer from a JSON string
void* tokenizers_new_from_str(const char* json);

// Free a tokenizer
void tokenizers_free(void* tokenizer);

// Encode text into token IDs
tokenizers_encoding_t tokenizers_encode(void* tokenizer, const char* text, bool add_special_tokens);

// Free an encoding
void tokenizers_free_encoding(tokenizers_encoding_t enc);

// Get tokenizer version
const char* tokenizers_version();

// Free a string returned by the library
void tokenizers_string_free(char* s);

// Get vocabulary size
size_t tokenizers_vocab_size(void* tokenizer);

// Get token ID for a token string
int tokenizers_token_to_id(void* tokenizer, const char* token);

// Add a special token
bool tokenizers_add_special_token(void* tokenizer, const char* token);

#ifdef __cplusplus
}
#endif

#endif // TOKENIZERS_C_H
