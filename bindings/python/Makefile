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
  CARGO_ENV := DYLD_FALLBACK_LIBRARY_PATH=$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
    PYTHONHOME=$(shell python3 -c "import sys; print(sys.base_prefix)")
else
  PIP := pip
  CARGO_ENV :=
endif

PYO3_PATCH := \n[patch.crates-io]\npyo3 = { git = "https://github.com/PyO3/pyo3.git", rev = "2ba9cda59a8b2fb07ad9b2b7f20d82e96d7ab0d2" }\npyo3-ffi = { git = "https://github.com/PyO3/pyo3.git", rev = "2ba9cda59a8b2fb07ad9b2b7f20d82e96d7ab0d2" }\n

# Format source code automatically
style:
	cp .cargo/config.toml .cargo/config.toml.bak
	printf '$(PYO3_PATCH)' >> .cargo/config.toml
	$(CARGO_ENV) cargo run --manifest-path ./tools/stub-gen/Cargo.toml; \
	  status=$$?; cp .cargo/config.toml.bak .cargo/config.toml; rm -f .cargo/config.toml.bak; exit $$status
	ruff check  $(check_dirs) --fix
	ruff format $(check_dirs)
	ty check py_src --exclude py_src/tokenizers/implementations --exclude py_src/tokenizers/tools/visualizer.py



# Check the source code is formatted correctly
check-style:
	cp .cargo/config.toml .cargo/config.toml.bak
	printf '$(PYO3_PATCH)' >> .cargo/config.toml
	$(CARGO_ENV) cargo run --manifest-path ./tools/stub-gen/Cargo.toml; \
	  status=$$?; cp .cargo/config.toml.bak .cargo/config.toml; rm -f .cargo/config.toml.bak; exit $$status
	ruff check $(check_dirs)
	ruff format py_src/tokenizers/*.pyi
	ruff format --check $(check_dirs)
	ty check py_src --exclude py_src/tokenizers/implementations --exclude py_src/tokenizers/tools/visualizer.py


TESTS_RESOURCES = $(DATA_DIR)/small.txt $(DATA_DIR)/roberta.json

# Launch the test suite
test: $(TESTS_RESOURCES)
	$(PIP) install pytest pytest-asyncio requests setuptools_rust numpy pyarrow datasets
	python -m pytest -s -v tests
	$(CARGO_ENV) cargo test --no-default-features

$(DATA_DIR)/big.txt :
	$(dir_guard)
	wget https://norvig.com/big.txt -O $@

$(DATA_DIR)/small.txt : $(DATA_DIR)/big.txt
	head -100 $(DATA_DIR)/big.txt > $@

$(DATA_DIR)/roberta.json :
	$(dir_guard)
	wget https://huggingface.co/roberta-large/raw/main/tokenizer.json -O $@
