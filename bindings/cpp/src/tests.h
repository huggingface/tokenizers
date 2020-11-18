#pragma once

#include "normalizers.h"
#include "pre_tokenizers.h"

#include <vector>
#include <string>
#include <memory>

namespace huggingface {
namespace tokenizers {
inline std::unique_ptr<std::vector<std::string>> run_tests() { return nullptr; }
}  // namespace tokenizers
}  // namespace huggingface
