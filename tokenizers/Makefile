DATA_DIR = data
BENCHMARK_DIR = benches
TESTS_DIR = tests

dir_guard=@mkdir -p $(@D)

SHARED_RESOURCES = $(DATA_DIR)/gpt2-vocab.json $(DATA_DIR)/gpt2-merges.txt $(DATA_DIR)/bert-base-uncased-vocab.txt $(DATA_DIR)/big.txt $(DATA_DIR)/small.txt $(DATA_DIR)/albert-base-v1-tokenizer.json  $(DATA_DIR)/llama-3-tokenizer.json
BENCHMARK_RESOURCES = $(SHARED_RESOURCES)
TESTS_RESOURCES = $(SHARED_RESOURCES) $(DATA_DIR)/unigram.json $(DATA_DIR)/unigram_wagahaiwa_nekodearu.txt $(DATA_DIR)/roberta.json $(DATA_DIR)/tokenizer-wiki.json $(DATA_DIR)/bert-wiki.json $(DATA_DIR)/tokenizer_config.json $(DATA_DIR)/bert_tokenizer_config.json

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
	wget https://huggingface.co/Narsil/small/raw/main/unigram$* -O $@

$(DATA_DIR)/albert-base-v1-tokenizer.json :
	$(dir_guard)
	wget https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-v1-tokenizer.json -O $@

$(DATA_DIR)/tokenizer-llama3.json :
	$(dir_guard)
	wget https://huggingface.co/Narsil/llama-tokenizer/resolve/main/tokenizer.json -O $@

$(DATA_DIR)/big.txt :
	$(dir_guard)
	wget https://norvig.com/big.txt -O $@

$(DATA_DIR)/small.txt : $(DATA_DIR)/big.txt
	head -100 $(DATA_DIR)/big.txt > $@

$(DATA_DIR)/roberta.json :
	$(dir_guard)
	wget https://huggingface.co/Narsil/small/raw/main/roberta.json -O $@

$(DATA_DIR)/tokenizer-wiki.json :
	$(dir_guard)
	wget https://s3.amazonaws.com/models.huggingface.co/bert/anthony/doc-quicktour/tokenizer.json -O $@

$(DATA_DIR)/bert-wiki.json :
	$(dir_guard)
	wget https://s3.amazonaws.com/models.huggingface.co/bert/anthony/doc-pipeline/tokenizer.json -O $@

$(DATA_DIR)/llama-3-tokenizer.json :
	$(dir_guard)
	wget https://huggingface.co/hf-internal-testing/llama3-tokenizer/resolve/main/tokenizer.json -O $@

# Config files for C++ bindings tests
$(DATA_DIR)/tokenizer_config.json :
	$(dir_guard)
	@echo '{"bos_token":"<bos>","eos_token":"<eos>","pad_token":"<pad>","unk_token":"<unk>","add_bos_token":true,"add_eos_token":false,"chat_template":"{% for message in messages %}{% if message['"'"'role'"'"'] == '"'"'user'"'"' %}{{ '"'"'User: '"'"' + message['"'"'content'"'"'] + '"'"'\\n'"'"' }}{% elif message['"'"'role'"'"'] == '"'"'assistant'"'"' %}{{ '"'"'Assistant: '"'"' + message['"'"'content'"'"'] + '"'"'\\n'"'"' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '"'"'Assistant: '"'"' }}{% endif %}"}' > $@

$(DATA_DIR)/bert_tokenizer_config.json :
	$(dir_guard)
	@echo '{"bos_token":"[CLS]","eos_token":"[SEP]","pad_token":"[PAD]","unk_token":"[UNK]","add_bos_token":true,"add_eos_token":true,"chat_template":null}' > $@
