.PHONY: develop stubs test examples style check-style

DATA_DIR = data
HF_TEST_REPO = hf-internal-testing/tokenizers-test-data
HF ?= hf
TESTS_RESOURCES = $(DATA_DIR)/tokenizer-wiki.json $(DATA_DIR)/gpt2.json $(DATA_DIR)/bert-wiki.json

# Debug build of the extension, installed editable into the active Python environment.
develop: stubs
	maturin develop

stubs:
	cargo run --manifest-path ./tools/stub-gen/Cargo.toml
	ruff format python/tokenizers/tokenizers.pyi

test: develop $(TESTS_RESOURCES)
	python -m pytest -s -v tests

examples: develop $(TESTS_RESOURCES)
	for example in examples/*.py; do python $$example; done

$(DATA_DIR)/%:
	mkdir -p $(DATA_DIR)
	$(HF) download $(HF_TEST_REPO) $* --repo-type dataset --local-dir $(DATA_DIR)

style: stubs
	cargo fmt
	ruff format python tests examples
	ruff check python tests examples --fix
	ty check python tests examples

check-style: stubs
	cargo fmt --check
	ruff format --check python tests examples
	ruff check python tests examples
	ty check python tests examples
