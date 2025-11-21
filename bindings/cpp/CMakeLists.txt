cmake_minimum_required(VERSION 3.16)
project(tokenizers_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Option to force a fresh cargo build
option(TOKENIZERS_CPP_FORCE_CARGO "Force rebuilding the Rust C FFI library" OFF)

# Build directory for Rust output (now at bindings/c)
set(RUST_CRATE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../c)
set(RUST_OUTPUT_DIR ${RUST_CRATE_DIR}/target/release)
set(RUST_LIB_NAME tokenizers_c)

# Custom command to build the Rust cdylib
add_custom_command(
    OUTPUT ${RUST_OUTPUT_DIR}/lib${RUST_LIB_NAME}.so
    WORKING_DIRECTORY ${RUST_CRATE_DIR}
    COMMAND cargo build --release
    COMMENT "Building Rust FFI crate tokenizers_c"
    DEPENDS ${RUST_CRATE_DIR}/src/lib.rs ${RUST_CRATE_DIR}/Cargo.toml
    VERBATIM
)

add_custom_target(build_rust_ffi DEPENDS ${RUST_OUTPUT_DIR}/lib${RUST_LIB_NAME}.so)

add_library(${RUST_LIB_NAME} SHARED IMPORTED GLOBAL)
add_dependencies(${RUST_LIB_NAME} build_rust_ffi)
set_target_properties(${RUST_LIB_NAME} PROPERTIES
    IMPORTED_LOCATION ${RUST_OUTPUT_DIR}/lib${RUST_LIB_NAME}.so
)

# C++ wrapper library
add_library(tokenizers_cpp INTERFACE)
add_dependencies(tokenizers_cpp build_rust_ffi)

target_include_directories(tokenizers_cpp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Tests
enable_testing()

# Single unified test executable
add_executable(tokenizer-tests 
    tests/main.cpp
    tests/test_common.cpp
    tests/test_basic.cpp
    tests/test_vocab_size.cpp
    tests/test_special_token_encode.cpp
    tests/test_encode_variations.cpp
    tests/test_error_handling.cpp
    tests/test_bert_tokenizer.cpp
    tests/test_serialization_decoding_batch.cpp
)
add_dependencies(tokenizer-tests build_rust_ffi)
target_link_libraries(tokenizer-tests PRIVATE ${RUST_LIB_NAME})
target_include_directories(tokenizer-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Register individual tests that invoke tokenizer-tests with different arguments
add_test(NAME tokenizers_cpp_basic COMMAND tokenizer-tests basic)
add_test(NAME tokenizers_cpp_vocab_size COMMAND tokenizer-tests vocab_size)
add_test(NAME tokenizers_cpp_special_token_encode COMMAND tokenizer-tests special_token_encode)
add_test(NAME tokenizers_cpp_encode_variations COMMAND tokenizer-tests encode_variations)
add_test(NAME tokenizers_cpp_error_handling COMMAND tokenizer-tests error_handling)
add_test(NAME tokenizers_cpp_bert_tokenizer COMMAND tokenizer-tests bert_tokenizer)
add_test(NAME tokenizers_cpp_serialization_decoding_batch COMMAND tokenizer-tests serialization_decoding_batch)

message(STATUS "tokenizers_cpp configured. Build with: cmake -S bindings/cpp -B build && cmake --build build && ctest --test-dir build")
