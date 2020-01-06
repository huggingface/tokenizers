#!/bin/bash
set -ex

# Create a symlink for tokenizers-lib
ln -sf ../../tokenizers tokenizers-lib
# Modify cargo.toml to include this symlink
sed -i 's/\.\.\/\.\.\/tokenizers/\.\/tokenizers-lib/' Cargo.toml
# Build the source distribution
python setup.py sdist
