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

HF_TEST_DATA = https://huggingface.co/datasets/hf-internal-testing/tokenizers-test-data/resolve/main

$(DATA_DIR)/gpt2-vocab.json :
	$(dir_guard)
	wget $(HF_TEST_DATA)/gpt2-vocab.json -O $@

$(DATA_DIR)/gpt2-merges.txt :
	$(dir_guard)
	wget $(HF_TEST_DATA)/gpt2-merges.txt -O $@

$(DATA_DIR)/bert-base-uncased-vocab.txt :
	$(dir_guard)
	wget $(HF_TEST_DATA)/bert-base-uncased-vocab.txt -O $@

$(DATA_DIR)/unigram.json :
	$(dir_guard)
	wget $(HF_TEST_DATA)/unigram.json -O $@

$(DATA_DIR)/unigram_wagahaiwa_nekodearu.txt :
	$(dir_guard)
	wget $(HF_TEST_DATA)/unigram_wagahaiwa_nekodearu.txt -O $@

$(DATA_DIR)/albert-base-v1-tokenizer.json :
	$(dir_guard)
	wget $(HF_TEST_DATA)/albert-base-v1-tokenizer.json -O $@

$(DATA_DIR)/big.txt :
	$(dir_guard)
	wget $(HF_TEST_DATA)/big.txt -O $@

$(DATA_DIR)/small.txt :
	$(dir_guard)
	wget $(HF_TEST_DATA)/small.txt -O $@

$(DATA_DIR)/roberta.json :
	$(dir_guard)
	wget $(HF_TEST_DATA)/roberta.json -O $@

$(DATA_DIR)/tokenizer-wiki.json :
	$(dir_guard)
	wget $(HF_TEST_DATA)/tokenizer-wiki.json -O $@

$(DATA_DIR)/bert-wiki.json :
	$(dir_guard)
	wget $(HF_TEST_DATA)/bert-wiki.json -O $@

$(DATA_DIR)/llama-3-tokenizer.json :
	$(dir_guard)
	wget $(HF_TEST_DATA)/llama-3-tokenizer.json -O $@
