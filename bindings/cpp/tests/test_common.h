#pragma once
#include <string>
#include <filesystem>
#include <cstdlib>

namespace test_utils {

inline std::string find_resource(const std::string& name) {
    namespace fs = std::filesystem;
    
    // First check environment variable (set by CMake or user)
    if (const char* env = std::getenv("TOKENIZERS_TEST_DATA")) {
        auto path = fs::path(env) / name;
        if (fs::exists(path)) return path.string();
    }
    
    // Fallback: search relative paths
    for (const auto& dir : {"data", "../data", "../../data", "../../../data"}) {
        auto path = fs::path(dir) / name;
        if (fs::exists(path)) return path.string();
    }
    return {};
}

} // namespace test_utils
