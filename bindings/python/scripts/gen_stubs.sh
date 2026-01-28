#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

echo "Building and installing extension (release, stub-gen enabled)..."
maturin develop --release --features stub-gen

echo "Refreshing cdylib used for introspection..."
cp target/release/libtokenizers.so tokenizers.abi3.so

echo "Generating stubs..."
cargo run --bin stub_generation --no-default-features --features stub-gen

echo "Done: tokenizers.pyi"
