.PHONY: style check-style test

DATA_DIR = data

dir_guard=@mkdir -p $(@D)

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
