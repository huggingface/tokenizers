.PHONY: style check-style

# Format source code automatically

style:
	black --line-length 100 --target-version py35 examples tokenizers tests

check-style:
	black --check --line-length 100 --target-version py35 examples tokenizers tests
