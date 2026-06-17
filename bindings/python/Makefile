.PHONY: style check-style test

DATA_DIR = data

dir_guard=@mkdir -p $(@D)
check_dirs := examples py_src/tokenizers tests
check_stubs := $(filter-out py_src/tokenizers/tokenizers.pyi,$(wildcard py_src/tokenizers/*.pyi py_src/tokenizers/*/*.pyi))

# Detect uv and set env vars to work around broken dylib install names
# in python-build-standalone distributions.
# See: https://github.com/astral-sh/uv/issues/11006
HAS_UV := $(shell command -v uv >/dev/null 2>&1 && echo 1 || echo 0)
ifeq ($(HAS_UV),1)
  PIP := uv pip
  CARGO_ENV := DYLD_FALLBACK_LIBRARY_PATH=$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
    PYTHONHOME=$(shell python3 -c "import sys; print(sys.base_prefix)")
else
  PIP := pip
  CARGO_ENV :=
endif

# Format source code automatically
style:
	$(CARGO_ENV) cargo run --manifest-path ./tools/stub-gen/Cargo.toml
	ruff check  $(check_stubs) --select I --fix
	ruff check  $(check_dirs) --fix
	ruff format $(check_dirs)
	ty check py_src --exclude py_src/tokenizers/implementations --exclude py_src/tokenizers/tools/visualizer.py



# Check the source code is formatted correctly
check-style:
	$(CARGO_ENV) cargo run --manifest-path ./tools/stub-gen/Cargo.toml
	ruff check $(check_stubs) --select I --fix
	ruff check $(check_dirs)
	ruff format py_src/tokenizers/*.pyi
	ruff format --check $(check_dirs)
	ty check py_src --exclude py_src/tokenizers/implementations --exclude py_src/tokenizers/tools/visualizer.py


TESTS_RESOURCES = $(DATA_DIR)/small.txt $(DATA_DIR)/roberta.json

# Launch the Python test suite
test-py: $(TESTS_RESOURCES)
	$(PIP) install pytest pytest-asyncio requests setuptools_rust numpy pyarrow datasets
	python -m pytest -s -v tests

# Launch the Rust test suite
# (cargo test --no-default-features uses pyo3's `auto-initialize` which
# links libpython; this doesn't work on free-threaded Python on macOS
# runners where libpython3.14t isn't installed in the framework path.)
test-rs:
	$(CARGO_ENV) cargo test --no-default-features

# Full test suite
test: test-py test-rs

HF_TEST_DATA = https://huggingface.co/datasets/hf-internal-testing/tokenizers-test-data/resolve/main

$(DATA_DIR)/big.txt :
	$(dir_guard)
	curl -sL $(HF_TEST_DATA)/big.txt -o $@

$(DATA_DIR)/small.txt :
	$(dir_guard)
	curl -sL $(HF_TEST_DATA)/small.txt -o $@

$(DATA_DIR)/roberta.json :
	$(dir_guard)
	curl -sL $(HF_TEST_DATA)/roberta.json -o $@
