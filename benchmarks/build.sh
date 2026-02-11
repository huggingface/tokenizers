#!/bin/bash
# Build script for all tokenizer variants

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Download big.txt if it doesn't exist
if [ ! -f "$SCRIPT_DIR/big.txt" ]; then
  echo ">>> Downloading big.txt..."
  curl -o "$SCRIPT_DIR/big.txt" https://norvig.com/big.txt
  echo "    ✓ big.txt downloaded"
  echo
fi


echo "=== Building all tokenizer variants ==="
echo

# Build Rust tokenizer
echo ">>> Building tokenizers-rust..."
cd "$ROOT_DIR/tokenizers"
cargo build --release --features http --example encode_batch
# Find the actual tokenizers rlib file
TOKENIZERS_LIB=$(find target/release/deps -name "libtokenizers-*.rlib" | head -n1)
if [ -z "$TOKENIZERS_LIB" ]; then
    echo "Error: Could not find tokenizers library file"
    exit 1
fi
rustc --edition 2018 -L target/release/deps -L target/release \
    --extern tokenizers="$TOKENIZERS_LIB" \
    "$SCRIPT_DIR/bench_rust.rs" \
    -o "$SCRIPT_DIR/bench_rust.out" \
    -C opt-level=3
echo "    ✓ Rust benchmark binary built"
echo

# Build Python bindings
echo ">>> Building tokenizers-python..."
cd "$ROOT_DIR/bindings/python"
pip install -e . --quiet || pip install -e .
chmod +x "$SCRIPT_DIR/bench_python.py"
echo "    ✓ Python bindings installed"
echo

# Build C bindings
echo ">>> Building tokenizers-c..."
cd "$ROOT_DIR/bindings/c"
cargo build --release
echo "    ✓ C bindings library built"
echo

# Build C benchmark binary
echo ">>> Building C benchmark..."
g++ -std=c++17 -O3 \
    -I"$ROOT_DIR/bindings/c" \
    "$SCRIPT_DIR/bench_c.cpp" \
    -o "$SCRIPT_DIR/bench_c.out" \
    -L"$ROOT_DIR/bindings/c/target/release" \
    -ltokenizers_c \
    -Wl,-rpath,"$ROOT_DIR/bindings/c/target/release"
echo "    ✓ C benchmark binary built"
echo

# Build C++ bindings
echo ">>> Building tokenizers-cpp bindings..."
cd "$ROOT_DIR/bindings/cpp"
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j$(nproc)
echo "    ✓ C++ bindings library built"
echo

# Build C++ benchmark binary
echo ">>> Building C++ benchmark..."
g++ -std=c++17 -O3 \
    -I"$ROOT_DIR/bindings/cpp/include" \
    "$SCRIPT_DIR/bench_cpp_bindings.cpp" \
    -o "$SCRIPT_DIR/bench_cpp_bindings.out" \
    -L"$ROOT_DIR/bindings/c/target/release" \
    -ltokenizers_c \
    -Wl,-rpath,"$ROOT_DIR/bindings/c/target/release"
echo "    ✓ C++ bindings benchmark binary built"
echo

echo "=== All builds completed successfully ==="
