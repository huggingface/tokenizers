#!/usr/bin/env bash
#
# Build the Python extension and run the binding tests against it.
#
# Why this exists as a script: `cargo check` proves the bindings *compile*, which is not the same as
# proving they still work. `PyModel`, `PyNormalizer`, `PyPreTokenizer`, `PyPostProcessor`,
# `PyDecoder` and `PyEncoding` all pickle by serialising through the wrapper enums
# (`#[serde(transparent)]` over `Arc<RwLock<…Wrapper>>`), so moving those wrappers between crates can
# keep the Rust compiling while silently changing what `pickle.dumps` emits — and pickling
# `tokenizers.models.BPE` is public API. The suite already carries ~60 pickle assertions across those
# six surfaces; this just makes sure they are actually run.
#
# Usage:
#   scripts/verify_bindings.sh                     # build a wheel, run the binding tests
#   scripts/verify_bindings.sh -k pickle           # any extra args go to pytest
#
# Needs `uv` (for an isolated interpreter) and `maturin`. Exits non-zero on the first failure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY_DIR="${REPO_ROOT}/bindings/python"

command -v uv >/dev/null || { echo "verify_bindings.sh: needs uv on PATH" >&2; exit 2; }

VENV="${TMPDIR:-/tmp}/tk-verify-venv"
if [ ! -x "${VENV}/bin/python" ]; then
  echo "==> creating an isolated venv at ${VENV}"
  uv venv "${VENV}" --python 3.13 >/dev/null
  VIRTUAL_ENV="${VENV}" uv pip install -q pytest pytest-asyncio numpy huggingface_hub maturin
fi

cd "${PY_DIR}"

echo "==> building the extension module"
VIRTUAL_ENV="${VENV}" maturin develop

# A stray `tokenizers.abi3.so` in this directory shadows the `py_src` package and makes every
# `from tokenizers.models import ...` fail with "not a package". maturin's stub generation leaves one
# behind, so move it aside rather than letting the failure look like a real regression.
if [ -f tokenizers.abi3.so ]; then
  echo "==> moving a stray tokenizers.abi3.so aside (it shadows the py_src package)"
  mv tokenizers.abi3.so "${TMPDIR:-/tmp}/stray-tokenizers.abi3.so"
fi

echo "==> import check"
"${VENV}/bin/python" - <<'PY'
import tokenizers
print(f"    tokenizers {tokenizers.__version__} from {tokenizers.__file__}")
PY

echo "==> binding tests"
# `test_tutorial_train_from_iterators` needs the `datasets` package and network; it is a docs
# tutorial, not a binding test.
"${VENV}/bin/python" -m pytest tests -q --no-header -p no:cacheprovider \
  --ignore=tests/documentation/test_tutorial_train_from_iterators.py "$@"
