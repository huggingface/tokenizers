DATA_DIR = data
BENCHMARK_DIR = benches
TESTS_DIR = tests

dir_guard=@mkdir -p $(@D)

SHARED_RESOURCES = $(DATA_DIR)/gpt2-vocab.json $(DATA_DIR)/gpt2-merges.txt $(DATA_DIR)/bert-base-uncased-vocab.txt $(DATA_DIR)/big.txt $(DATA_DIR)/small.txt
BENCHMARK_RESOURCES = $(SHARED_RESOURCES)
TESTS_RESOURCES = $(SHARED_RESOURCES) $(DATA_DIR)/unigram.json $(DATA_DIR)/unigram_wagahaiwa_nekodearu.txt $(DATA_DIR)/albert-base-v1-tokenizer.json $(DATA_DIR)/roberta.json $(DATA_DIR)/tokenizer-wiki.json $(DATA_DIR)/bert-wiki.json

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

$(DATA_DIR)/gpt2-% :
	$(dir_guard)
	wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-$* -O $@

$(DATA_DIR)/bert-% :
	$(dir_guard)
	wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-$* -O $@

$(DATA_DIR)/unigram% :
	$(dir_guard)
	wget https://storage.googleapis.com/tokenizers/unigram$* -O $@

$(DATA_DIR)/albert-base-v1-tokenizer.json :
	$(dir_guard)
	wget https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-v1-tokenizer.json -O $@

$(DATA_DIR)/big.txt :
	$(dir_guard)
	wget https://norvig.com/big.txt -O $@

$(DATA_DIR)/small.txt : $(DATA_DIR)/big.txt
	head -100 $(DATA_DIR)/big.txt > $@

$(DATA_DIR)/roberta.json :
	$(dir_guard)
	wget https://storage.googleapis.com/tokenizers/roberta.json -O $@

$(DATA_DIR)/tokenizer-wiki.json :
	$(dir_guard)
	wget https://s3.amazonaws.com/models.huggingface.co/bert/anthony/doc-quicktour/tokenizer.json -O $@

$(DATA_DIR)/bert-wiki.json :
	$(dir_guard)
	wget https://s3.amazonaws.com/models.huggingface.co/bert/anthony/doc-pipeline/tokenizer.json -O $@
