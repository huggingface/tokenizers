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

# Run tests (Google Test suite)
ctest --test-dir build-cpp -V
```

## FFI API Surface

C++ `Tokenizer` class methods:
- `load(path)` / constructor - load tokenizer from JSON file
- `FromBlobJSON(json)` - load tokenizer from JSON string (static method)
- `encode(text, add_special_tokens=true)` - encode text to token IDs
- `encode_batch(texts, add_special_tokens=true)` - encode batch of texts
- `decode(ids, skip_special_tokens=true)` - decode IDs to string
- `decode_batch(batch_ids, skip_special_tokens=true)` - decode batch of IDs
- `vocab_size()` - get vocabulary size
- `token_to_id(token)` - lookup token ID (returns -1 if not found)
- `id_to_token(id)` - lookup token string (returns empty if not found)
- `add_special_token(token)` - add a special token to vocabulary
- `add_special_tokens(tokens)` - add multiple special tokens
- `set_padding(params)` - configure padding
- `disable_padding()` - disable padding
- `set_truncation(params)` - configure truncation
- `disable_truncation()` - disable truncation
- `save(path, pretty=true)` - save tokenizer to JSON file
- `to_string(pretty=false)` - serialize tokenizer to JSON string
- `valid()` - check if tokenizer loaded successfully
- `version()` - get FFI version string (static method)

## Test Coverage

C++ binding tests are now unified using Google Test in `bindings/cpp/tests/test_tokenizer_gtest.cpp`.
The suite covers:
- Basic encode/decode
- Batch encode/decode
- Vocabulary operations
- Padding and Truncation
- Special tokens management
- Serialization (save/load/to_string)
- Error handling
- Integration with BERT tokenizer

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
