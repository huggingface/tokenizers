.PHONY: style check-style test

DATA_DIR = data

dir_guard=@mkdir -p $(@D)
check_dirs := examples py_src/tokenizers tests

# Detect uv and set env vars to work around broken dylib install names
# in python-build-standalone distributions.
# See: https://github.com/astral-sh/uv/issues/11006
HAS_UV := $(shell command -v uv >/dev/null 2>&1 && echo 1 || echo 0)
ifeq ($(HAS_UV),1)
  PIP := uv pip
  CARGO_TEST_ENV := DYLD_FALLBACK_LIBRARY_PATH=$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
    PYTHONHOME=$(shell python3 -c "import sys; print(sys.base_prefix)")
else
  PIP := pip
  CARGO_TEST_ENV :=
endif

# Format source code automatically
style:
	cargo run --manifest-path tools/stub-gen/Cargo.toml
	python stub.py
	ruff check  $(check_dirs) --fix
	ruff format $(check_dirs)
	ty check py_src --exclude py_src/tokenizers/implementations

# Check the source code is formatted correctly
check-style:
	python stub.py --check
	ruff check $(check_dirs)
	ruff format --check $(check_dirs)
	ty check py_src --exclude py_src/tokenizers/implementations

TESTS_RESOURCES = $(DATA_DIR)/small.txt $(DATA_DIR)/roberta.json

# Launch the test suite
test: $(TESTS_RESOURCES)
	$(PIP) install pytest pytest-asyncio requests setuptools_rust numpy pyarrow datasets
	python -m pytest -s -v tests
	$(CARGO_TEST_ENV) cargo test --no-default-features

$(DATA_DIR)/big.txt :
	$(dir_guard)
	wget https://norvig.com/big.txt -O $@

$(DATA_DIR)/small.txt : $(DATA_DIR)/big.txt
	head -100 $(DATA_DIR)/big.txt > $@

$(DATA_DIR)/roberta.json :
	$(dir_guard)
	wget https://huggingface.co/roberta-large/raw/main/tokenizer.json -O $@
