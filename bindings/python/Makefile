PYTHON := .venv/bin/python

# Everything needed to hack on the bindings: venv, deps, release build.
.PHONY: dev
dev: .venv
	. .venv/bin/activate && maturin develop --release

# huggingface_hub backs from_pretrained (the wheel's optional `hub` extra);
# the pretrained tests and example 01 need it.
.venv:
	uv venv .venv
	uv pip install --python $(PYTHON) maturin numpy pytest ruff mypy huggingface_hub

# Regenerate the .pyi stubs from the built extension. Run after `make dev`.
.PHONY: stubs
stubs:
	cargo run --manifest-path tools/stub-gen/Cargo.toml

.PHONY: test
test: dev
	$(PYTHON) -m pytest -q
	$(PYTHON) -m mypy.stubtest tokenizers --allowlist stubtest_allowlist.txt

# Examples 01-03 read this corpus; fetch it through the tokenizers repo's
# data rule, pointing it at the dev venv's `hf` CLI so nothing global is
# needed.
TOKENIZERS_DIR := ../../tokenizers
$(TOKENIZERS_DIR)/data/big.txt: | .venv
	$(MAKE) -C $(TOKENIZERS_DIR) data/big.txt HF=$(abspath .venv/bin/hf)

# Run the end-to-end examples. `datasets` (for 06) is installed on demand so
# `make dev` stays lean; 06 downloads wikitext-2 (~12 MB) on first run.
.PHONY: examples
examples: dev $(TOKENIZERS_DIR)/data/big.txt
	uv pip install --python $(PYTHON) datasets
	$(PYTHON) examples/01_pretrained.py
	$(PYTHON) examples/02_train_and_encode.py
	$(PYTHON) examples/03_threading.py
	$(PYTHON) examples/04_train_bert_wordpiece.py
	$(PYTHON) examples/05_train_bytelevel_bpe.py
	$(PYTHON) examples/06_train_with_datasets.py

# End-to-end tests (tests/e2e): the transformers scenario suite, two builds.
#   e2e          — the in-tree build; failures list what the rewrite still misses
#   e2e-release  — the tokenizers wheel transformers resolves; must always pass
# Dedicated venvs: transformers+torch are heavy, and each venv carries its own
# tokenizers. The dev venv also needs transformers' import-time tokenizers pin
# lifted, or it refuses our 1.0.0-dev version (see relax_transformers_pin.py).
.venv-e2e:
	uv venv $@ --python 3.12
	uv pip install --python $@/bin/python -r tests/e2e/requirements.txt maturin
	$@/bin/python tests/e2e/relax_transformers_pin.py

.PHONY: e2e
e2e: .venv-e2e
	. .venv-e2e/bin/activate && maturin develop --release
	.venv-e2e/bin/python -m pytest tests/e2e -q -ra --continue-on-collection-errors

.venv-e2e-release:
	uv venv $@ --python 3.12
	uv pip install --python $@/bin/python -r tests/e2e/requirements.txt

.PHONY: e2e-release
e2e-release: .venv-e2e-release
	.venv-e2e-release/bin/python -m pytest tests/e2e -q

# Golden conformance layer (tests/golden): exact encode/decode outputs on the
# data/ fixtures, diffed against committed goldens. The goldens are only ever
# regenerated from the released wheel (golden-regen) — never hand-edited.
# `golden` is opt-in (env gate) while the rewrite is incomplete; drop the gate
# once it runs green so plain pytest enforces conformance everywhere.
.PHONY: golden
golden: dev
	TOKENIZERS_GOLDEN=1 $(PYTHON) -m pytest tests/golden -q -ra

.PHONY: golden-regen
golden-regen: .venv .release
	PYTHONPATH=.release $(PYTHON) tests/golden/generate.py

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
	rm -rf .venv .venv-e2e .venv-e2e-release .release target tools/stub-gen/target
