.PHONY: build test test-data vite react style check-style check-types

DATA_DIR = data
HF_TEST_REPO = hf-internal-testing/tokenizers-test-data
HF ?= hf
TESTS_RESOURCES = $(DATA_DIR)/gpt2.json $(DATA_DIR)/llama-3-tokenizer.json $(DATA_DIR)/bert-base-uncased.json $(DATA_DIR)/t5-base.json

# Firefox by default: wasm-pack downloads the newest chromedriver, which refuses an older installed Chrome.
# For Chrome: `make test BROWSER=chrome`, adding `CHROMEDRIVER=<path>` when the versions differ.
BROWSER ?= firefox
CHROMEDRIVER ?=
WASM_PACK_TEST_FLAGS = --headless --$(BROWSER) $(if $(CHROMEDRIVER),--chromedriver $(CHROMEDRIVER))

NPM_INSTALL = npm install --no-audit --no-fund
NPM_CI = npm ci --no-audit --no-fund

# Extra cargo flags, e.g. `CARGO_FLAGS=--locked` in CI.
CARGO_FLAGS ?=

# ES module package in pkg/, loadable from a browser without a bundler.
build:
	wasm-pack build --release --target web -- $(CARGO_FLAGS)

test-data: $(TESTS_RESOURCES)

test: test-data
	wasm-pack test $(WASM_PACK_TEST_FLAGS) -- $(CARGO_FLAGS)

$(DATA_DIR)/%:
	mkdir -p $(DATA_DIR)
	$(HF) download $(HF_TEST_REPO) $* --repo-type dataset --local-dir $(DATA_DIR)

style:
	cargo fmt
	$(NPM_INSTALL) && npm run format

# Clippy compiles the tests, which embed the fixtures.
check-style: test-data
	cargo fmt --check
	cargo clippy $(CARGO_FLAGS) --all-targets -- -D warnings
	$(NPM_CI) && npm run lint

# Tsify generates the TypeScript declarations, so check them where they land.
check-types: build
	$(NPM_CI) && npm run typecheck

react: build
	cd examples/react && $(NPM_INSTALL) && npm run dev
