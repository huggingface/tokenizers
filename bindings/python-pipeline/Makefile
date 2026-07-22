PYTHON := .venv/bin/python

# Everything needed to hack on the bindings: venv, deps, release build.
.PHONY: dev
dev: .venv
	. .venv/bin/activate && maturin develop --release

.venv:
	uv venv .venv
	uv pip install --python $(PYTHON) maturin numpy tokenizers

# Regenerate the .pyi stubs from the built extension. Run after `make dev`.
.PHONY: stubs
stubs:
	cargo run --manifest-path tools/stub-gen/Cargo.toml

# Run the end-to-end examples (train, pretrained parity, threading).
.PHONY: examples
examples: dev
	$(PYTHON) examples/01_train_and_encode.py
	$(PYTHON) examples/02_pretrained.py
	$(PYTHON) examples/03_threading.py

# Benchmark against the released tokenizers wheel (same script CI runs).
.PHONY: bench
bench: dev
	$(PYTHON) benches/bench_vs_release.py

.PHONY: lint
lint:
	cargo fmt --check
	cargo clippy --all-targets -- -D warnings

.PHONY: clean
clean:
	rm -rf .venv target tools/stub-gen/target
