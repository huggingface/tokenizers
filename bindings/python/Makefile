.PHONY: style check-style test

DATA_DIR = data

dir_guard=@mkdir -p $(@D)

# Format source code automatically
style:
	python stub.py
	black --line-length 100 --target-version py35 examples py_src/tokenizers tests

# Check the source code is formatted correctly
check-style:
	python stub.py --check
	black --check --line-length 100 --target-version py35 examples py_src/tokenizers tests

TESTS_RESOURCES = $(DATA_DIR)/small.txt $(DATA_DIR)/roberta.json

# Launch the test suite
test: $(TESTS_RESOURCES)
	python -m pytest -s -v tests
	cargo test --no-default-features

$(DATA_DIR)/big.txt :
	$(dir_guard)
	wget https://norvig.com/big.txt -O $@

$(DATA_DIR)/small.txt : $(DATA_DIR)/big.txt
	head -100 $(DATA_DIR)/big.txt > $@

$(DATA_DIR)/roberta.json :
	$(dir_guard)
	wget https://storage.googleapis.com/tokenizers/roberta.json -O $@
