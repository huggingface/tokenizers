#pragma once

#include "tokenizers-cpp/tokens.rs.h"
#include "rust/cxx.h"

namespace rust {
// TODO https://github.com/dtolnay/cxx/issues/551
template <>
struct IsRelocatable<huggingface::tokenizers::Token> : std::true_type {};

template <>
struct IsRelocatable<huggingface::tokenizers::Tokens> : std::true_type {};
}  // namespace rust
