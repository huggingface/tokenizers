# C++ Bindings Example

This example demonstrates how to use the HuggingFace Tokenizers C++ bindings.

## Building

```bash
# Make sure test resources are available (includes sample tokenizer JSON files)
make -C tokenizers test

# Build the example
cmake -S bindings/cpp/example -B build_example
cmake --build build_example

# Run the example with a tokenizer file
./build_example/tokenizer_example ../../tokenizers/data/tokenizer.json "Hello world!"
```

## What This Example Shows

The example program demonstrates:

1. **Basic Encoding**: Encoding text to token IDs with and without special tokens
2. **Token Lookup**: Looking up token IDs by token string
3. **Adding Special Tokens**: Dynamically adding custom special tokens to the vocabulary
4. **Batch Processing**: Encoding multiple texts efficiently
5. **Move Semantics**: Using C++11 move semantics for efficient resource management
6. **Error Handling**: Checking tokenizer validity and handling missing tokens

## Usage

```bash
# Basic usage with default text
./build_example/tokenizer_example <path_to_tokenizer.json>

# Encode custom text
./build_example/tokenizer_example <path_to_tokenizer.json> "Your custom text here"
```

## Example Output

```
Tokenizers C++ Bindings Version: tokenizers_c 0.0.1

Loading tokenizer from: ../../tokenizers/data/tokenizer.json
✓ Tokenizer loaded successfully

Vocabulary size: 30000

=== Example 1: Basic Encoding ===
Input text: "Hello world!"
Tokens (with special tokens): [79, 33, 56, 63, 63, 66, 88, 66, 69, 63, 55, 5]
Token count: 12

=== Example 2: Encoding Without Special Tokens ===
Tokens (without special tokens): [79, 33, 56, 63, 63, 66, 88, 66, 69, 63, 55]
Token count: 11

...
```

## Integration into Your Project

To use the tokenizers C++ bindings in your own CMake project:

```cmake
# Add tokenizers as a subdirectory
add_subdirectory(path/to/tokenizers/bindings/cpp ${CMAKE_BINARY_DIR}/tokenizers_build)

# Link your target
target_link_libraries(your_target PRIVATE tokenizers_cpp tokenizers_c)
target_include_directories(your_target PRIVATE path/to/tokenizers/bindings/cpp/include)
```

Then in your C++ code:

```cpp
#include "tokenizers/tokenizers.h"
using namespace tokenizers;

Tokenizer tok("path/to/tokenizer.json");
auto ids = tok.encode("Hello world!");
```
