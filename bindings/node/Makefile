.PHONY: style check-style test

DATA_DIR = data

dir_guard=@mkdir -p $(@D)

HF_TEST_REPO = hf-internal-testing/tokenizers-test-data
# Fetch fixtures through the huggingface_hub CLI: it handles auth (HF_TOKEN),
# retry/backoff on rate limits, and caching for us. CI overrides HF with
# `uvx --from huggingface_hub hf` to avoid a separate install step.
HF ?= hf

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

# small.txt is derived locally; everything else is fetched from the Hub by name.
$(DATA_DIR)/small.txt : $(DATA_DIR)/big.txt
	head -100 $(DATA_DIR)/big.txt > $@

$(DATA_DIR)/% :
	$(dir_guard)
	$(HF) download $(HF_TEST_REPO) $* --repo-type dataset --local-dir $(DATA_DIR)
