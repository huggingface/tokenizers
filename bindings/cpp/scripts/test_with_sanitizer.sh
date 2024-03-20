#!/bin/sh

set -eu

SANITIZER="${1:-address}"

cd $(dirname "$(readlink -f "$0")")/..

RUSTFLAGS="-Zsanitizer=$SANITIZER" CFLAGS="-fsanitize=$SANITIZER" CXXFLAGS="-fsanitize=$SANITIZER" LDFLAGS="-fsanitize=$SANITIZER" cargo +nightly test --features=test --target x86_64-unknown-linux-gnu
