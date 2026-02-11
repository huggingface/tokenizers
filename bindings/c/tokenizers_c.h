#ifndef TOKENIZERS_C_H
#define TOKENIZERS_C_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Only define the struct if not already defined
#ifndef TOKENIZERS_ENCODING_T_DEFINED
#define TOKENIZERS_ENCODING_T_DEFINED
typedef struct {
    const int* ids;
    const int* attention_mask;
    size_t len;
    void* _internal_ptr;  // Internal use only - do not access
} tokenizers_encoding_t;
#endif

// Create a new tokenizer from a JSON file (auto-loads tokenizer_config.json if present)
void* tokenizers_new_from_file(const char* path);

// Create a new tokenizer with explicit config file path
void* tokenizers_new_from_file_with_config(const char* path, const char* config_path);

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

// Get token string for a token ID
char* tokenizers_id_to_token(void* tokenizer, int id);

// Decode token IDs back to text
char* tokenizers_decode(void* tokenizer, const int* ids, size_t len, bool skip_special_tokens);

// Add a special token
bool tokenizers_add_special_token(void* tokenizer, const char* token);

// === Special Tokens (unified API) ===
// Config is auto-loaded from tokenizer_config.json if present next to tokenizer.json

// Get special token ID by name ("BOS", "EOS", "PAD", "UNK")
// Uses config if available, falls back to heuristic. Returns -1 if not found.
int tokenizers_get_special_token_id(void* tokenizer, const char* name);

// Get special token string by name ("BOS", "EOS", "PAD", "UNK")
// Returns token from config, or NULL if not available. Must free with tokenizers_string_free.
char* tokenizers_get_special_token(void* tokenizer, const char* name);

// Get add_bos_token setting (false if no config)
bool tokenizers_get_add_bos_token(void* tokenizer);

// Get add_eos_token setting (false if no config)
bool tokenizers_get_add_eos_token(void* tokenizer);

// Check if tokenizer has a chat template
bool tokenizers_has_chat_template(void* tokenizer);

// Get chat template string (must be freed with tokenizers_string_free)
char* tokenizers_get_chat_template(void* tokenizer);

// Apply a chat template to render messages
// Arguments:
//   - tokenizer: the tokenizer instance
//   - template_str: Jinja2 template string
//   - messages_json: JSON array of messages with "role" and "content" fields
//   - add_generation_prompt: whether to append generation prompt
//   - bos_token: optional BOS token string (can be NULL)
//   - eos_token: optional EOS token string (can be NULL)
//   - error_out: pointer to error string (caller must free with tokenizers_string_free)
// Returns: rendered template string (caller must free with tokenizers_string_free), or NULL on error
char* tokenizers_apply_chat_template(
    void* tokenizer,
    const char* template_str,
    const char* messages_json,
    bool add_generation_prompt,
    const char* bos_token,
    const char* eos_token,
    char** error_out
);

#ifdef __cplusplus
}
#endif

#endif // TOKENIZERS_C_H
