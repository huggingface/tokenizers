# C++ Bindings for HuggingFace Tokenizers

Minimal C++17 wrapper over the Rust `tokenizers` crate.

## Quick Start

See the [example project](example/) for a complete, working demonstration of all features.

```bash
# Build and run the example
cmake -S bindings/cpp/example -B build_example
cmake --build build_example
./build_example/tokenizer_example path/to/tokenizer.json "Your text here"
```

## Overview

Architecture:
- Rust FFI crate (`tokenizers_c`) exposes a C ABI (load, encode, vocab ops, special tokens).
- Header-only C++ class `tokenizers::Tokenizer` provides RAII, `encode()` returning `std::vector<int32_t>`.
- Build system: CMake + cargo. CTest for tests.

## Build

Prerequisites: Rust toolchain, CMake >= 3.16, a C++17 compiler.

```bash

# prerequisite 1: Install rustc and cargo, if you dont have it already
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env" 

# NOTE: the below commands should be run from the tokenizers repo root

# prerequisite 2: original tokenizer (rust) can be built and tested
make -C ./tokenizers test

# Configure & build
cmake -S bindings/cpp -B build-cpp
cmake --build build-cpp -j
# if you run out of memory, replace "-j" (use all cores) with "-j4" (use only 4 cores)

# Run tests (6 C++ binding tests + original Rust test suite)
ctest --test-dir build-cpp -V
```

## FFI API Surface

C++ `Tokenizer` class methods:
- `load(path)` / constructor - load tokenizer from JSON file
- `FromBlobJSON(json)` - load tokenizer from JSON string (static method)
- `encode(text, add_special_tokens=true)` - encode text to token IDs
- `encode_batch(texts, add_special_tokens=true)` - encode batch of texts
- `decode(ids, skip_special_tokens=true)` - decode IDs to string
- `vocab_size()` - get vocabulary size
- `token_to_id(token)` - lookup token ID (returns -1 if not found)
- `id_to_token(id)` - lookup token string (returns empty if not found)
- `add_special_token(token)` - add a special token to vocabulary
- `add_special_tokens(tokens)` - add multiple special tokens
- `save(path, pretty=true)` - save tokenizer to JSON file
- `to_string(pretty=false)` - serialize tokenizer to JSON string
- `valid()` - check if tokenizer loaded successfully
- `version()` - get FFI version string (static method)

## Test Coverage

C++ binding tests (`bindings/cpp/tests`):
1. **test_basic** - Basic encode/decode smoke test
2. **test_vocab_size** - Vocab size growth after adding special tokens
3. **test_special_token_encode** - Special token encoding validation
4. **test_encode_variations** - Encode with/without special tokens, empty input, consistency
5. **test_error_handling** - Invalid file loading, move semantics, nonexistent tokens
6. **test_bert_tokenizer** - BERT tokenizer integration with multiple texts
7. **test_new_features** - Test new APIs (decode, id_to_token, save, to_string, encode_batch, add_special_tokens)

Original Rust tests also available via `ctest -R tokenizers_rust_all`.

## Usage

Add `bindings/cpp/include` to your include path and link against the generated `libtokenizers_c.so` (or platform equivalent) built in `bindings/c/target/release`.

Example:
```cpp
#include "tokenizers/tokenizers.h"
using namespace tokenizers;

int main() {
    Tokenizer tok("path/to/tokenizer.json");
    if (!tok.valid()) return 1;
    
    auto ids = tok.encode("Hello world!");
    for (auto id : ids) {
        std::cout << id << " ";
    }
    
    std::string decoded = tok.decode(ids);
    std::cout << "\nDecoded: " << decoded << "\n";
}
```

## Notes & Future Improvements
- Error handling returns empty/default values; could be extended with status codes/exceptions.
- Full Rust test suite available through CTest for integration tracking.
- Thread safety: Create one instance per thread or add mutex.

## License
Apache-2.0 (same as upstream project).
