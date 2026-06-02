.PHONY: style check-style test

DATA_DIR = data

dir_guard=@mkdir -p $(@D)

HF_DATA = https://huggingface.co/datasets/hf-internal-testing/tokenizers-test-data/resolve/main

# Format source code automatically
style:
	npm run lint

# Check the source code is formatted correctly
check-style:
	npm run lint-check

TESTS_RESOURCES = $(DATA_DIR)/small.txt $(DATA_DIR)/roberta.json $(DATA_DIR)/tokenizer-wiki.json $(DATA_DIR)/bert-wiki.json

# Launch the test suite
test: $(TESTS_RESOURCES)
	npm run test

$(DATA_DIR)/big.txt :
	$(dir_guard)
	curl -sL $(HF_DATA)/big.txt -o $@

$(DATA_DIR)/small.txt : $(DATA_DIR)/big.txt
	head -100 $(DATA_DIR)/big.txt > $@

$(DATA_DIR)/roberta.json :
	$(dir_guard)
	curl -sL $(HF_DATA)/roberta.json -o $@

$(DATA_DIR)/tokenizer-wiki.json :
	$(dir_guard)
	curl -sL $(HF_DATA)/tokenizer-wiki.json -o $@

$(DATA_DIR)/bert-wiki.json :
	$(dir_guard)
	curl -sL $(HF_DATA)/bert-wiki.json -o $@
