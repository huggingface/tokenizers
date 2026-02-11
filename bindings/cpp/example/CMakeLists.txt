cmake_minimum_required(VERSION 3.16)
project(tokenizers_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include the tokenizers C++ bindings as a subdirectory
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_BINARY_DIR}/tokenizers_cpp_build)

# Example executable
add_executable(tokenizer_example main.cpp)
target_link_libraries(tokenizer_example PRIVATE tokenizers_cpp tokenizers_c)
target_include_directories(tokenizer_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

message(STATUS "Example project configured. Build with: cmake -S bindings/cpp/example -B build_example && cmake --build build_example")
