#pragma once

#define DOCTEST_CONFIG_IMPLEMENT

#include "normalizers.h"
#include "pre_tokenizers.h"
#include "tokenizers-cpp/src/tests.rs.h"
#include "rust/cxx.h"

#include <doctest/doctest.h>

#include <vector>
#include <string>
#include <memory>

namespace huggingface {
namespace tokenizers {
inline bool run_tests() {
    doctest::Context context;
    return context.run() == 0;
}
}  // namespace tokenizers
}  // namespace huggingface
