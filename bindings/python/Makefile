.PHONY: style check-style test

DATA_DIR = data

dir_guard=@mkdir -p $(@D)
check_dirs := examples py_src/tokenizers tests

# Format source code automatically
style:
	python stub.py
	ruff check  $(check_dirs) --fix 
	ruff format $(check_dirs)

# Check the source code is formatted correctly
check-style:
	python stub.py --check
	ruff check $(check_dirs)
	ruff format --check $(check_dirs)

TESTS_RESOURCES = $(DATA_DIR)/small.txt $(DATA_DIR)/roberta.json

# Launch the test suite
test: $(TESTS_RESOURCES)
	pip install pytest requests setuptools_rust numpy pyarrow datasets
	python -m pytest -s -v tests
	cargo test --no-default-features

$(DATA_DIR)/big.txt :
	$(dir_guard)
	wget https://norvig.com/big.txt -O $@

$(DATA_DIR)/small.txt : $(DATA_DIR)/big.txt
	head -100 $(DATA_DIR)/big.txt > $@

$(DATA_DIR)/roberta.json :
	$(dir_guard)
	wget https://huggingface.co/roberta-large/raw/main/tokenizer.json -O $@
