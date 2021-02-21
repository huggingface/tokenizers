.PHONY: style check-style test

DATA_DIR = data

dir_guard=@mkdir -p $(@D)

# Format source code automatically
style:
	npm run lint

# Check the source code is formatted correctly
check-style:
	npm run lint-check

TESTS_RESOURCES = $(DATA_DIR)/small.txt $(DATA_DIR)/roberta.json

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
