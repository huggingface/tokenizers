#include "test_common.h"
#include <filesystem>
#include <vector>

namespace test_utils {

std::string find_resource(const std::string& name) {
    // data directory is linked to rust project's data directory
    // run "make -C ../../tokenizers test" i.e. point -C to rust project depending on where make is run from 
    namespace fs = std::filesystem;
    std::vector<fs::path> candidates = {
        fs::path("./data") / name,
        fs::path("../data") / name,
        fs::path("../../data") / name,
        fs::path("../../../data") / name,
    };
    for (auto& c : candidates) {
        if (fs::exists(c)) return c.string();
    }
    return {};
}

} // namespace test_utils
