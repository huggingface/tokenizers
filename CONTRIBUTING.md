# Contributing to Tokenizers

## Repository layout

```
tokenizers/
  tokenizers/         # Core Rust library (the main crate)
    src/              # Library source code
    benches/          # Criterion benchmarks
    Makefile          # Build, test, bench, lint targets (with auto data download)
  bindings/
    python/           # PyO3 Python bindings
      Makefile        # Python-specific test and lint targets
    node/             # Node.js bindings
  docs/               # Documentation
```

The core Rust crate lives in `tokenizers/` (not the repo root). Most `make` and
`cargo` commands need to be run from that subdirectory.

## Prerequisites

- **Rust** (stable): install via [rustup](https://rustup.rs/)
- **Python 3.9+**: for the Python bindings
- **wget**: used by the Makefile to download test/benchmark data (`brew install wget` on macOS)
- **maturin**: for building the Python bindings (`pip install maturin`)

## Getting started

### 1. Clone and set up a Python environment

```bash
git clone https://github.com/huggingface/tokenizers.git
cd tokenizers

# Create a virtualenv (using uv, venv, or your preferred tool)
python -m venv .venv
source .venv/bin/activate
```

### 2. Build and test the Rust core

```bash
cd tokenizers
make test    # downloads test data automatically via wget, then runs cargo test
```

The first run downloads model files and corpora into `tokenizers/data/`. These
files are gitignored.

### 3. Build and test the Python bindings

```bash
cd bindings/python
pip install -e ".[dev]"   # install in editable mode with test deps (builds via maturin)
make test                 # run pytest, then cargo test
```

If you need to rebuild after Rust changes without reinstalling:

```bash
pip install maturin       # if not already installed
maturin develop           # fast rebuild of the extension module
```

### 4. Run benchmarks

```bash
cd tokenizers
make bench   # downloads benchmark data if needed, then runs cargo bench
```

Benchmark results are stored in `target/criterion/` for comparison across runs.
To run a specific benchmark:

```bash
cargo bench --bench bpe_benchmark
cargo bench --bench bert_benchmark
cargo bench --bench llama3_benchmark
cargo bench --bench layout_benchmark
```

## Known issues

### uv-managed Python and `cargo test` on macOS

If you use [uv](https://github.com/astral-sh/uv) to manage Python, `cargo test`
for the Python bindings may fail with:

```
Library not loaded: /install/lib/libpython3.X.dylib
```

This is a [known issue](https://github.com/astral-sh/uv/issues/11006) with
uv's prebuilt Python distributions — the shared library has a broken install
name on macOS. The `bindings/python/Makefile` detects uv and applies the
workaround automatically when you use `make test`. If you run `cargo test`
directly, set these environment variables first:

```bash
export DYLD_FALLBACK_LIBRARY_PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
export PYTHONHOME="$(python3 -c 'import sys; print(sys.base_prefix)')"
cargo test --no-default-features
```

### Benchmark data not found

The benchmarks expect data files in `tokenizers/data/`. These are **not** checked
into the repository. Running `make bench` or `make test` from the `tokenizers/`
directory will download them automatically. If you prefer to run `cargo bench`
directly, download the data first:

```bash
cd tokenizers
make data/big.txt data/gpt2-vocab.json data/gpt2-merges.txt   # etc.
```

Or download all benchmark data at once:

```bash
make bench   # will fetch everything before running benchmarks
```

## Development workflows

### Formatting and linting

```bash
# Rust core
cd tokenizers
make lint      # rustfmt --check + clippy

# Python bindings
cd bindings/python
make style         # auto-format
make check-style   # check formatting
```

### Running a subset of tests

```bash
# Rust core — specific test
cd tokenizers
cargo test test_name

# Python bindings — specific test file
cd bindings/python
python -m pytest tests/bindings/test_tokenizer.py -v -k "test_name"

# Python bindings — Rust-side tests only
cargo test --no-default-features
```

### Profiling

For performance work, [samply](https://github.com/mstange/samply) is useful for
generating CPU profiles:

```bash
cd tokenizers
cargo build --release --example my_bench
samply record ./target/release/examples/my_bench
```

Profile from Python to see the full stack including PyO3 overhead:

```bash
samply record python my_script.py
```
