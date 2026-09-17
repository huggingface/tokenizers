DATA_DIR = data
BENCHMARK_DIR = benches
TESTS_DIR = tests

dir_guard=@mkdir -p $(@D)

SHARED_RESOURCES = $(DATA_DIR)/gpt2-vocab.json $(DATA_DIR)/gpt2-merges.txt $(DATA_DIR)/bert-base-uncased-vocab.txt $(DATA_DIR)/big.txt $(DATA_DIR)/small.txt $(DATA_DIR)/albert-base-v1-tokenizer.json  $(DATA_DIR)/llama-3-tokenizer.json
BENCHMARK_RESOURCES = $(SHARED_RESOURCES)
TESTS_RESOURCES = $(SHARED_RESOURCES) $(DATA_DIR)/unigram.json $(DATA_DIR)/unigram_wagahaiwa_nekodearu.txt $(DATA_DIR)/roberta.json $(DATA_DIR)/tokenizer-wiki.json $(DATA_DIR)/bert-wiki.json

.PHONY : build
build :
	cargo build --all-targets

.PHONY : release
release :
	cargo build --release

.PHONY : format
format :
	cargo fmt --

.PHONY : lint
lint :
	cargo fmt -- --check
	cargo fmt -- $(BENCHMARK_DIR)/*.rs --check
	cargo clippy --all-targets --all-features -- -D warnings

.PHONY : test
test : $(TESTS_RESOURCES)
	cargo test

.PHONY : doc
doc :
	cargo doc

.PHONY : publish
publish :
	cargo publish

.PHONY : all-checks
all-checks : lint test doc

.PHONY : bench
bench : $(BENCHMARK_RESOURCES)
	cargo bench -- --verbose

HF_TEST_REPO = hf-internal-testing/tokenizers-test-data
# Fetch fixtures through the huggingface_hub CLI: it handles auth (HF_TOKEN),
# retry/backoff on rate limits, and caching for us. CI overrides HF with
# `uvx --from huggingface_hub hf` to avoid a separate install step.
HF ?= hf

$(DATA_DIR)/% :
	$(dir_guard)
	$(HF) download $(HF_TEST_REPO) $* --repo-type dataset --local-dir $(DATA_DIR)
