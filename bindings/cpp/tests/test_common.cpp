#include "test_common.h"
#include <filesystem>
#include <vector>

namespace test_utils {

std::string find_resource(const std::string& name) {
    std::vector<std::filesystem::path> candidates = {
        std::filesystem::path("../tokenizers/data") / name,
        std::filesystem::path("../../tokenizers/data") / name,
        std::filesystem::path("tokenizers/data") / name,
        std::filesystem::path("./data") / name
    };
    for (auto& c : candidates) {
        if (std::filesystem::exists(c)) return c.string();
    }
    return {};
}

} // namespace test_utils
