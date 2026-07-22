PYTHON := .venv/bin/python

# Everything needed to hack on the bindings: venv, deps, release build.
.PHONY: dev
dev: .venv
	. .venv/bin/activate && maturin develop --release

.venv:
	uv venv .venv
	uv pip install --python $(PYTHON) maturin numpy pytest ruff mypy

# Regenerate the .pyi stubs from the built extension. Run after `make dev`.
.PHONY: stubs
stubs:
	cargo run --manifest-path tools/stub-gen/Cargo.toml

.PHONY: test
test: dev
	$(PYTHON) -m pytest -q
	$(PYTHON) -m mypy.stubtest tokenizers --allowlist stubtest_allowlist.txt

# Run the end-to-end examples. `datasets` (for 06) is installed on demand so
# `make dev` stays lean; 06 downloads wikitext-2 (~12 MB) on first run.
.PHONY: examples
examples: dev
	uv pip install --python $(PYTHON) datasets
	$(PYTHON) examples/01_pretrained.py
	$(PYTHON) examples/02_train_and_encode.py
	$(PYTHON) examples/03_threading.py
	$(PYTHON) examples/04_train_bert_wordpiece.py
	$(PYTHON) examples/05_train_bytelevel_bpe.py
	$(PYTHON) examples/06_train_with_datasets.py

# The released PyPI wheel shares our package name, so it lives in its own
# directory that the bench subprocess puts on PYTHONPATH.
.release: | .venv
	uv pip install --python $(PYTHON) --target .release tokenizers

# Benchmark against the released tokenizers wheel (same script CI runs).
.PHONY: bench
bench: dev .release
	$(PYTHON) benches/bench_vs_release.py

.PHONY: lint
lint: .venv
	cargo fmt --check
	cargo clippy --all-targets -- -D warnings
	.venv/bin/ruff check benches examples tests
	.venv/bin/ruff format --check benches examples tests

.PHONY: clean
clean:
	rm -rf .venv .release target tools/stub-gen/target
