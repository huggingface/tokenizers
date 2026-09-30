.PHONY: build test test-package test-data vite react style check-style check-types

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
# The js/ entry points initialize the wasm module on import, so users never call init().
build:
	wasm-pack build --release --target web -- $(CARGO_FLAGS)
	cp js/* pkg/
	cd pkg && npm pkg set main=index.js types=index.d.ts 'files[]=index.js' 'files[]=node.js' 'files[]=index.d.ts' 'sideEffects[]=./index.js' 'sideEffects[]=./node.js' && \
	npm pkg set --json exports='{".": {"types": "./index.d.ts", "node": "./node.js", "default": "./index.js"}}'

test-data: $(TESTS_RESOURCES)

test: test-data
	wasm-pack test $(WASM_PACK_TEST_FLAGS) -- $(CARGO_FLAGS)

# Installs the packed tarball in a scratch project, as a user would, and round-trips a text through it.
test-package: build test-data
	tmp=$$(mktemp -d) && trap 'rm -rf "$$tmp"' EXIT && \
	npm pack ./pkg --pack-destination "$$tmp" && \
	cp tests/package.mjs "$$tmp" && \
	cd "$$tmp" && npm init -y >/dev/null && $(NPM_INSTALL) ./*.tgz && \
	node package.mjs "$$(node -p 'require("$(CURDIR)/pkg/package.json").name')" "$(CURDIR)/$(DATA_DIR)/gpt2.json"

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
