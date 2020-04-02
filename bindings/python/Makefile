.PHONY: style check-style test

# Format source code automatically
style:
	black --line-length 100 --target-version py35 examples tokenizers tests

# Check the source code is formatted correctly
check-style:
	black --check --line-length 100 --target-version py35 examples tokenizers tests

# Launch the test suite
test:
	python -m pytest -s -v tests
