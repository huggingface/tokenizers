# ---- fixtures ----

DATA_DIR = data
dir_guard=@mkdir -p $(@D)

HF_TEST_REPO = hf-internal-testing/tokenizers-test-data
# CI overrides with `uvx --from huggingface_hub hf`.
HF ?= hf

HF_REVISION ?= a6c4e9fcdc6e18f7ae3ef21253a28443e538c714

$(DATA_DIR)/% :
	$(dir_guard)
	$(HF) download $(HF_TEST_REPO) $* --repo-type dataset --revision $(HF_REVISION) --local-dir $(DATA_DIR)

SHARED_RESOURCES = $(DATA_DIR)/gpt2-vocab.json $(DATA_DIR)/gpt2.json $(DATA_DIR)/big.txt $(DATA_DIR)/albert-base-v1-tokenizer.json $(DATA_DIR)/llama-3-tokenizer.json
BENCHMARK_RESOURCES = $(SHARED_RESOURCES) $(DATA_DIR)/unigram_wagahaiwa_nekodearu.txt $(DATA_DIR)/deepseek-v4.json
TESTS_RESOURCES = $(SHARED_RESOURCES) $(DATA_DIR)/unigram.json $(DATA_DIR)/unigram_wagahaiwa_nekodearu.txt $(DATA_DIR)/roberta.json $(DATA_DIR)/tokenizer-wiki.json $(DATA_DIR)/bert-wiki.json
ORACLE_RESOURCES = $(DATA_DIR)/gpt2.json $(DATA_DIR)/bert-base-uncased.json $(DATA_DIR)/t5-base.json $(DATA_DIR)/albert-base-v1-tokenizer.json $(DATA_DIR)/fixtures/models/llama-3.2-1b.json $(DATA_DIR)/fixtures/models/all-minilm-l6-v2.json $(DATA_DIR)/fixtures/models/all-mpnet-base-v2.json $(DATA_DIR)/fixtures/models/siglip-base-patch16-224.json $(DATA_DIR)/fixtures/models/mistral-7b-v0.1.json

.PHONY : data
data : $(TESTS_RESOURCES) $(BENCHMARK_RESOURCES)

# ---- dev loop ----

.PHONY : build
build :
	cargo build --workspace --all-targets

.PHONY : release
release :
	cargo build --workspace --release

.PHONY : format
format :
	cargo fmt --all --

.PHONY : lint
lint :
	cargo fmt --all -- --check
	cargo clippy --workspace --all-targets --all-features -- -D warnings

.PHONY : test
test : $(TESTS_RESOURCES)
	cargo test --workspace --no-fail-fast

# Parity against the latest released `tokenizers` crate, on HF_REVISION-pinned fixtures.
.PHONY : oracle
oracle : $(ORACLE_RESOURCES)
	cargo test -p tk-convert --features bench-baseline --test oracle

.PHONY : doc
doc :
	cargo doc --workspace

.PHONY : publish
publish :
	cargo publish

# ---- aggregate checks ----

# `cfg` gates on enum variants, so a combination can break on its own.
MODEL_FEATURES = bpe,unigram,wordpiece,wordlevel,normalizers

.PHONY : all-checks
all-checks : lint test doc feature-matrix oracle

# tk-encode under each feature combination; `lint` and `test` are blind to these. Needs cargo-hack.
.PHONY : feature-matrix
feature-matrix : $(SHARED_RESOURCES)
	cargo hack --package tk-encode --feature-powerset --depth 2 \
	  --include-features $(MODEL_FEATURES) clippy --all-targets -- -D warnings
	cargo hack --package tk-encode --each-feature \
	  --include-features $(MODEL_FEATURES) test --lib

# ---- size profiling (opt-in, not part of all-checks) ----

# Gzipped only: Mach-O segments are 16 KiB-quantised, so on-disk size cannot resolve small changes.
.PHONY : slim-size
slim-size :
	cargo build --profile minsize -p tk-serialize --example binsize_pipeline --no-default-features --features bpe,deserialize
	@gzip -9 -c target/minsize/examples/binsize_pipeline | wc -c | tr -d ' ' \
	  | xargs printf 'slim tk-encode + tk-serialize: %s bytes gzipped\n'

# Explicit triple required: `-Z build-std` is a no-op without an explicit --target.
HOST_TRIPLE := $(shell rustc -vV | awk '/^host:/ {print $$2}')
# Overridable against a stale default: `make slimest NIGHTLY=nightly-2026-04-28`.
NIGHTLY ?= nightly
SLIMEST_FLAGS = -Z build-std=std,panic_abort --target $(HOST_TRIPLE)

# RUSTFLAGS, not `panic = "immediate-abort"` in the profile: stable cargo cannot parse that key.
SLIMEST_RUSTFLAGS = -Zunstable-options -Cpanic=immediate-abort

# `minsize` plus a `std` rebuilt without panic/unwinding; needs nightly + rust-src.
.PHONY : slimest
slimest :
	RUSTFLAGS="$(SLIMEST_RUSTFLAGS)" cargo +$(NIGHTLY) build --profile slimest $(SLIMEST_FLAGS) \
	  -p tk-serialize --example binsize_pipeline --no-default-features --features bpe,deserialize

# Same features as `slim-size` on purpose, so the two numbers differ by the profile alone.
.PHONY : slimest-size
slimest-size : slimest
	@b=target/$(HOST_TRIPLE)/slimest/examples/binsize_pipeline; \
	  gzip -9 -c $$b | wc -c | tr -d ' ' | xargs printf 'slimest tk-encode: %s bytes gzipped\n'

# ---- bench ----

# Cross-engine benchmarks live in huggingface/tokbench.
.PHONY : bench
bench : $(BENCHMARK_RESOURCES)
	cargo bench -p tk-serialize
