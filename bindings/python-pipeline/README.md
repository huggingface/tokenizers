# tokenizers-pipeline

Experimental Python bindings for 🤗 tokenizers, built on the `PipelineTokenizer`
encode path. Same `tokenizer.json` files, same ids, and much faster through
Python: encode never holds the GIL, batches run multi-threaded in Rust, inputs
are borrowed instead of copied, and ids come back as `numpy.uint32` arrays
without a copy.

```python
import tokenizers_pipeline as tp

tok = tp.Tokenizer.from_file("tokenizer.json")
ids = tok.encode("Hello world", add_special_tokens=False)   # np.ndarray[uint32]
batch = tok.encode_batch(lines, add_special_tokens=False)   # list of arrays
```

Training and in-place modification work too:

```python
tok = tp.Tokenizer(tp.models.BPE())
tok.normalizer = tp.normalizers.Lowercase()
tok.pre_tokenizer = tp.pre_tokenizers.Whitespace()
tok.train_from_iterator(lines, trainer=tp.trainers.BpeTrainer(vocab_size=30000))
tok.save("tokenizer.json")
```

Not there yet (loud errors, never wrong ids): `decode`, post-processor
templates (`[CLS]`/`<s>` insertion — pass `add_special_tokens=False`), and the
`Metaspace` pre-tokenizer (t5-style files).

## Build and use locally

Requirements: Rust (stable), [uv](https://docs.astral.sh/uv/), Python ≥ 3.10.

```sh
cd bindings/python-pipeline
make dev          # venv + deps + release build, installed editable
source .venv/bin/activate
python -c "import tokenizers_pipeline; print(tokenizers_pipeline.__version__)"
```

Rebuild after changing Rust code with `make dev` again (or `maturin develop
--release` inside the venv). Always use `--release`: a debug build encodes
10-100× slower and any timing you take from it is meaningless.

To build a distributable wheel instead: `maturin build --release` (find it in
`target/wheels/`).

Other targets:

```sh
make examples     # run the three end-to-end examples (needs ../../tokenizers/data)
make bench        # benchmark against the released tokenizers wheel
make stubs        # regenerate the .pyi type stubs from the built extension
make lint         # cargo fmt --check + clippy -D warnings
```

The examples and the benchmark read test data from `../../tokenizers/data`.
Fetch it once with `make -C ../../tokenizers fixtures bench-models data/big.txt`
(needs `HF_TOKEN` for the mirror repo).

## Type stubs are generated

Do not edit the `.pyi` files under `py_src/` by hand. They are produced by
`tools/stub-gen`, which reads the introspection metadata pyo3 embeds in the
built extension — so run `make dev` first, then `make stubs`. Docstrings and
signatures come from the Rust sources; return types that introspection cannot
see (numpy arrays, `Self`) are declared with
`#[pyo3(signature = (...) -> "Type")]` annotations in the Rust code.

## How it works

A `Tokenizer` holds two things behind one lock:

- the **spec** — the plain Rust `Tokenizer`, the serializable source of truth.
  Setters, `train*`, and `add_*` write here.
- the **compiled pipeline** — an immutable `Arc<PipelineTokenizer>` the encode
  methods share with worker threads. Any mutation drops it; the next encode
  rebuilds it once. Configurations the pipeline cannot run fail at that point
  with the reason, never with different ids.

Every method releases the GIL before touching the lock — enforced at compile
time by `DetachedRwLock` (see `src/detached_lock.rs`), with a clippy ban on
`Python::attach` as the backstop.

## Benchmark

`benches/bench_vs_release.py` times `encode_batch` end-to-end through Python
against the latest released `tokenizers` wheel, on the same corpora and ~10 KiB
chunking as the Rust benchmark (`tk-encode/examples/fixture_bench.rs`): every
fixture under `data/fixtures/{lang,modalities}`, warmed up, median of N runs,
single-thread per fixture plus one multi-thread sweep, ids verified equal
before timing. CI runs it in the `python-bindings-bench` job of the Pipeline
Benchmark workflow, posts the table to the run's step summary, and the report
job renders it as a chart (`.github/scripts/render_python_bench.py`) appended
to the benchmark section in the PR description, next to the Rust charts.
