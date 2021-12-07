set -e
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORK_DIR
RUST_MANIFEST="./src/main/rust/Cargo.toml"
cargo build --manifest-path $RUST_MANIFEST --release
cargo test --features c-headers  --manifest-path $RUST_MANIFEST -- generate_headers

