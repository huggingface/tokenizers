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

# Jinja2Cpp for chat template rendering
set(JINJA2CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(JINJA2CPP_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(JINJA2CPP_DEPS_MODE internal CACHE STRING "" FORCE)
add_subdirectory(third_party/Jinja2Cpp)

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

# C++ wrapper library with chat template support
add_library(tokenizers_cpp_impl STATIC
    src/tokenizers.cpp
)
add_dependencies(tokenizers_cpp_impl build_rust_ffi)
target_include_directories(tokenizers_cpp_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(tokenizers_cpp_impl PUBLIC ${RUST_LIB_NAME} jinja2cpp)

# Interface library for easy linking
add_library(tokenizers_cpp INTERFACE)
target_link_libraries(tokenizers_cpp INTERFACE tokenizers_cpp_impl)

# Tests
enable_testing()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Google Test executable
add_executable(tokenizer_tests_gtest
    tests/test_tokenizer_gtest.cpp
)
target_link_libraries(tokenizer_tests_gtest PRIVATE tokenizers_cpp GTest::gtest_main)

# Set test data directory for test discovery
set(TOKENIZERS_TEST_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data")

# Register Google Test with environment variable for test data
include(GoogleTest)
gtest_discover_tests(tokenizer_tests_gtest
    PROPERTIES ENVIRONMENT "TOKENIZERS_TEST_DATA=${TOKENIZERS_TEST_DATA_DIR}"
)

message(STATUS "tokenizers_cpp configured. Build with: cmake -S bindings/cpp -B build && cmake --build build && ctest --test-dir build")
