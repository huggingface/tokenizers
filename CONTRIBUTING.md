# Contributing to Tokenizers

## Opening an issue

**An issue without a reproducer will be closed.** "The tokenizer is wrong" or
"this is slow" is not a bug report. We need:

- **An actual `tokenizer.json`** — a Hub repo id we can download, or the file
  attached. Not a description of it, not a screenshot, not a snippet of the
  vocab.
- **The exact input text** and the ids you got, next to the ids you expected.
- **A concrete use case.** What are you actually doing? Which model, which
  pipeline, what breaks downstream? Abstract or hypothetical reports get
  closed — we cannot fix a tokenizer we cannot load.
- Versions: `tokenizers`, Python/Node, OS and CPU architecture.

Short runnable snippet, please. If we cannot paste it into a terminal and see
your bug, it is not a reproducer.

## Opening a pull request

**You are responsible for the code you push.** If you cannot explain every line
of your diff, why it is correct, and what it breaks if it is wrong — do not open
it as ready for review.

- **No AI slop.** Using a model to help is fine. Opening a PR you have not read,
  do not understand, and cannot defend is not. Plausible-looking diffs that no
  human has thought about cost us more time than the bug did, and they get
  closed without a review.
- **Keep it a draft** until you are genuinely committed to understanding the
  codebase and proposing a proper fix. A draft is free. A PR marked ready for
  review is a claim that you have done the work.
- **Fix the cause, not the symptom.** A patch that makes your case pass while
  leaving the underlying bug in place will not be merged.
- **Tests come with the fix.** A bug fix without a regression test is
  incomplete.
- Keep the diff focused. Unrelated reformatting, renames and drive-by
  refactors make a change unreviewable.

### Claiming a performance improvement

Anything claiming to be faster must come with numbers, and the numbers must come
from [tokbench](https://github.com/huggingface/tokbench). Run it on your own
machine, before and after, and paste both results into the PR along with your
CPU model and OS.

We will not merge a speedup on the strength of a microbenchmark you wrote for
your own patch, or on reasoning about why it should be faster. Tokenizer
performance is dominated by cache behaviour and input script — changes that look
obviously good are routinely neutral or negative once measured across the
matrix. Byte-exactness of output ids is non-negotiable: a faster tokenizer that
moves a single id is a broken tokenizer.

## Repository layout

```
tokenizers/
  tokenizers/               # Rust workspace root (NOT the repo root)
    src/                    # The `tokenizers` umbrella crate
    tk-encode/              # Inference: model engines, pipeline components, Tokenizer
    tk-serialize/           # Reader for canonical tokenizer.json (no serde)
      benches/              # Criterion benchmarks (encode, decode)
    tk-convert/             # Legacy tokenizer.json -> canonical JSON upgrade pass
    tk-train/               # Training: Trainer trait, the *Trainer types
    bitcannon/              # SIMD Unicode classification + bitstream pre-tokenization
    bitmap_gen/             # Dev tool: regenerates bitcannon's classify tables
    scripts/                # Helper scripts (e.g. verify_bindings.sh)
    data/                   # Test/bench fixtures, downloaded on demand (gitignored)
    Makefile                # build, test, lint, bench, size targets
  bindings/
    python/                 # PyO3 bindings, built with maturin
    node/                   # Node.js bindings (napi)
  docs/                     # Documentation sources
```

The Rust workspace lives in `tokenizers/`, one level below the repo root. Most
`make` and `cargo` commands must be run from there.

`tk-train` is excluded from the workspace, so `cargo test --workspace` does not
cover it — build it explicitly if you touch it.

## Setup

### Prerequisites

- **Rust** (stable), via [rustup](https://rustup.rs/)
- **[uv](https://docs.astral.sh/uv/)** — used for Python, the `hf` CLI and
  maturin. Everything below assumes it:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### Rust core

```bash
git clone https://github.com/huggingface/tokenizers.git
cd tokenizers/tokenizers

make test    # downloads fixtures into data/, then cargo test --workspace
```

Fixtures come from the Hub through the `hf` CLI. If you do not want it installed
globally, hand the Makefile uv's throwaway runner — this is exactly what CI does:

```bash
make test HF="uvx --from huggingface_hub hf"
```

Fixtures are pinned to a revision (`HF_REVISION` in the Makefile), so they do not
move under you. `make data` fetches everything without running anything.

### Python bindings

```bash
cd bindings/python

uv sync                 # creates .venv and installs the dev group (maturin, pytest, ruff, ty)
source .venv/bin/activate
make develop            # regenerates the .pyi stubs, then `maturin develop`
make test               # develop + fixtures + pytest
```

After a Rust change, `maturin develop` rebuilds the extension module in place —
no reinstall needed. Add `--release` when you are benchmarking; the default build
is a debug build and is far slower than the shipped one.

`make style` formats (cargo fmt, ruff, stub regeneration) and `make check-style`
verifies without writing. Stubs are generated — edit the Rust signatures, not
`tokenizers.pyi`.

### Node bindings

```bash
cd bindings/node
yarn install && yarn build && yarn test
```

## Development workflows

### Tests and lint

```bash
cd tokenizers

make test           # whole workspace
make lint           # rustfmt --check + clippy -D warnings
make all-checks     # lint + test + doc + feature matrix
cargo test test_name
```

`make feature-matrix` compiles and tests `tk-encode` under each meaningful
feature combination — the per-model and `normalizers` `cfg` gates interact, and
neither `lint` (`--all-features`) nor `test` (default features) can see the
builds in between. It needs `cargo install cargo-hack`.

### Benchmarks

```bash
cd tokenizers
make bench   # fetches benchmark data, then cargo bench -p tk-serialize
```

Results land in `target/criterion/` for comparison across runs. Individually:

```bash
cargo bench -p tk-serialize --bench encode
cargo bench -p tk-serialize --bench decode
```

`encode` reports the fused pipeline plus one row per stage (added-token scan,
normalize, pre-tokenize, model), so a regression can be attributed to a stage.
The stage rows are single-threaded and take a `&str`, so they do not sum to the
fused rows — the gap is the fusion, the threading and the post-processor.

The cross-engine comparisons, the model matrix and the per-language sweeps live
in [tokbench](https://github.com/huggingface/tokbench). That is the benchmark a
performance PR has to move.

### Binary size

```bash
cd tokenizers
make slim-size      # stripped + gzipped tk-encode under --profile minsize
make slimest-size   # same, plus a std rebuilt without panic/unwinding (needs nightly + rust-src)
```

Gzipped is the only honest number: Mach-O segments are 16 KiB-quantised, so
on-disk size cannot resolve changes smaller than that. Neither target is part of
`all-checks` — there is no agreed size budget, so they only print numbers.

### Profiling

[samply](https://github.com/mstange/samply) for CPU profiles:

```bash
cd tokenizers
cargo build --release --example my_bench
samply record ./target/release/examples/my_bench
```

From Python, to see the full stack including PyO3 overhead:

```bash
samply record python my_script.py
```

## Known issues

### uv-managed Python and `cargo test` on macOS

`cargo test` for the Python bindings may fail with:

```
Library not loaded: /install/lib/libpython3.X.dylib
```

This is a [known issue](https://github.com/astral-sh/uv/issues/11006) with uv's
prebuilt Python distributions — the shared library has a broken install name on
macOS. `bindings/python/Makefile` detects uv and works around it, so `make test`
is fine. Running `cargo test` directly needs:

```bash
export DYLD_FALLBACK_LIBRARY_PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
export PYTHONHOME="$(python3 -c 'import sys; print(sys.base_prefix)')"
cargo test --no-default-features
```

### Fixtures not found

Benchmarks and tests read from `tokenizers/data/`, which is not checked in.
`make test` and `make bench` download what they need. To fetch everything up
front:

```bash
cd tokenizers
make data
```
