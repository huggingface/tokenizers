.PHONY: develop stubs test examples style check-style

# Fetched from the Hub on demand, same fixture repo `bindings/node` uses.
DATA_DIR = data
HF_TEST_REPO = hf-internal-testing/tokenizers-test-data
HF ?= hf
TESTS_RESOURCES = $(DATA_DIR)/tokenizer-wiki.json $(DATA_DIR)/gpt2.json $(DATA_DIR)/bert-wiki.json

# Editable install: builds the extension in debug mode and installs it into
# the active Python environment. Fast inner loop; does not touch the .pyi
# stubs (use `make stubs` after changing a signature or docstring).
develop:
	maturin develop

# Regenerates python/tokenizers/tokenizers.pyi from a release build of the extension: the
# stub-gen tool builds it, then reads the signatures and docstrings pyo3 embedded in the
# binary, so the stub can never drift from what's exported.
stubs:
	cargo run --manifest-path ./tools/stub-gen/Cargo.toml

test: develop $(TESTS_RESOURCES)
	python -m pytest -s -v tests

examples: develop $(TESTS_RESOURCES)
	for example in examples/*.py; do python $$example; done

$(DATA_DIR)/%:
	mkdir -p $(DATA_DIR)
	$(HF) download $(HF_TEST_REPO) $* --repo-type dataset --local-dir $(DATA_DIR)

# pyo3-introspection's own output isn't ruff-formatted, so it's reformatted in place
# unconditionally rather than checked; everything else only gets checked, never rewritten
# for you silently.
GENERATED_STUBS := $(wildcard python/tokenizers/*.pyi)

style: stubs
	cargo fmt
	ruff format $(GENERATED_STUBS)
	ruff format python tests examples
	ruff check python tests examples --fix
	ty check python tests examples

check-style: stubs
	cargo fmt --check
	ruff format $(GENERATED_STUBS)
	ruff format --check python tests examples
	ruff check python tests examples
	ty check python tests examples
