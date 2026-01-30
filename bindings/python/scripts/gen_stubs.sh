#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

echo "Building and installing extension (release, stub-gen enabled)..."
maturin develop --release --features stub-gen

echo "Refreshing cdylib used for introspection..."
if [ "$(uname)" == "Darwin" ]; then
    cp target/release/libtokenizers.dylib tokenizers.abi3.so
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    cp target/release/libtokenizers.so tokenizers.abi3.so
fi


echo "Generating stubs..."
cargo run --bin stub_generation --no-default-features --features stub-gen

echo "Done: tokenizers.pyi"
