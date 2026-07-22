# tokenizers (Python bindings)

Tokenizers turn text into the sequences of integer ids that language models
consume. This package is the Python interface to Hugging Face's Rust
[tokenizers](https://github.com/huggingface/tokenizers) library.

This is the 1.x rewrite of the bindings: it loads the same `tokenizer.json`
files as 0.x and produces the same ids, but is much faster through Python —
encoding runs in Rust threads without blocking your Python program, and ids
come back as ready-to-use `numpy` arrays.

```python
import tokenizers as tk

tok = tk.Tokenizer.from_file("tokenizer.json")

# Returns a numpy.uint32 array of token ids.
# add_special_tokens=False skips template tokens like [CLS]/[SEP]; inserting
# them is not implemented yet in 1.0, so leaving it True raises a loud
# NotImplementedError on tokenizers that use such templates (BERT, Llama, …).
ids = tok.encode("Hello world", add_special_tokens=False)

# A list of arrays, encoded in parallel across Rust threads.
batch = tok.encode_batch(["Hello world", "How are you?"], add_special_tokens=False)
```

To load a tokenizer straight from the [Hugging Face Hub](https://huggingface.co),
install the `hub` extra (`pip install 'tokenizers[hub]'`):

```python
tok = tk.Tokenizer.from_pretrained("openai-community/gpt2")
```

## Training your own tokenizer

A tokenizer has three parts, applied in order:

- a **normalizer** cleans the text (lowercasing, Unicode fix-ups),
- a **pre-tokenizer** cuts it into pieces (usually words),
- the **model** turns each piece into ids, using a vocabulary learned during
  training (BPE, WordPiece, Unigram, or WordLevel).

You pick the parts, then train the model's vocabulary on your own text:

```python
tok = tk.Tokenizer(tk.models.BPE())
tok.normalizer = tk.normalizers.Lowercase()
tok.pre_tokenizer = tk.pre_tokenizers.Whitespace()
tok.train_from_iterator(lines, trainer=tk.trainers.BpeTrainer(vocab_size=30000))
tok.save("tokenizer.json")
```

The `examples/` directory walks through all of this, starting with the
simplest case (load a pretrained file and encode).

## Threading and async

Encoding releases the Python interpreter lock (the GIL), so this package
plays well with threads and event loops:

- Calling `encode` from several Python threads scales — the threads really
  run in parallel, on every interpreter (free-threaded or not).
- `encode_batch` parallelizes one batch across Rust threads. Set the
  `TOKENIZERS_PARALLELISM` environment variable to `false`/`true` to
  disable/force this.
- In `asyncio` code, `await tok.async_encode(text)` /
  `await tok.async_encode_batch(texts)` keep the event loop free while Rust
  encodes in a worker thread.

```python
ids = await tok.async_encode("Hello world", add_special_tokens=False)
```

## Breaking changes vs 0.x

1.x is a ground-up rewrite with a smaller, faster API. The headline changes:

- `encode` returns a `numpy.uint32` array of ids, not an `Encoding` object.
  Tokens, offsets, type ids, attention masks, and word ids are gone from the
  encode path, and so are truncation and padding
  (`enable_truncation`/`enable_padding`).
- `encode` takes a single text: no `pair=` argument, no `is_pretokenized=`.
- Not implemented yet (loud errors, never wrong ids): `decode`,
  post-processor templates (`[CLS]`/`<s>` insertion — pass
  `add_special_tokens=False`), and the `Metaspace` pre-tokenizer
  (t5-style files).
- Custom Python components (normalizers/pre-tokenizers written in Python)
  are not supported; components are plain values you assign, not objects
  you subclass.
- `decoders`, `processors`, and the `implementations` helpers
  (`BertWordPieceTokenizer`, …) are gone.
- **`transformers` cannot use 1.0 as its backend yet** — it needs several of
  the removed pieces. Pin `tokenizers<1.0` for `transformers`.

The full list, including smaller removals and renames, is in the 1.0.0 entry
of [CHANGELOG.md](CHANGELOG.md).

Kept from 0.x: `async_encode`/`async_encode_batch` and free-threaded Python
support. New in 1.0: parity-aware BPE training across several languages
(`trainers.ParityBpeTrainer`).

## Build and use locally

Requirements: Rust (stable), [uv](https://docs.astral.sh/uv/), Python ≥ 3.10.

```sh
cd bindings/python
make dev          # venv + deps + release build, installed editable
source .venv/bin/activate
python -c "import tokenizers; print(tokenizers.__version__)"
```

Rebuild after changing Rust code with `make dev` again (or `maturin develop
--release` inside the venv). Always use `--release`: a debug build encodes
10-100× slower and any timing you take from it is meaningless.

To build a distributable wheel instead: `maturin build --release` (find it in
`target/wheels/`). Default wheels use the stable Python ABI (abi3): one
binary per platform covers CPython 3.10–3.14. Free-threaded interpreters
(3.13t/3.14t) cannot load abi3 extensions, so their wheels are built
per-version with `maturin build --no-default-features`.

Other targets:

```sh
make test         # pytest suite in tests/
make examples     # run the end-to-end examples (needs ../../tokenizers/data)
make bench        # benchmark against the released tokenizers wheel from PyPI
make stubs        # regenerate the .pyi type stubs from the built extension
make lint         # cargo fmt + clippy, ruff over the python sources
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

CI enforces this twice: the committed stubs must match what stub-gen
produces, and `mypy.stubtest` checks them against the actual runtime
(accepted differences are listed and explained in `stubtest_allowlist.txt`).

## How it works (internals)

A `Tokenizer` holds two things behind one lock:

- the **spec** — the plain Rust `Tokenizer`, the serializable source of truth.
  Setters, `train*`, and `add_*` write here.
- the **compiled pipeline** — an immutable `Arc<PipelineTokenizer>` the encode
  methods share with worker threads. Any mutation drops it; the next encode
  rebuilds it once. Configurations the pipeline cannot run fail at that point
  with the reason, never with different ids.

Every method releases the GIL before touching the lock — enforced at compile
time by `DetachedRwLock` (see `src/detached_lock.rs`), with a clippy ban on
`Python::attach` as the backstop. Input strings are borrowed, not copied, and
the output arrays take ownership of the Rust buffers, also copy-free.

## Benchmark

`benches/bench_vs_release.py` times `encode_batch` end-to-end through Python
against the latest released `tokenizers` wheel, on the same corpora and ~10 KiB
chunking as the Rust benchmark (`tk-encode/examples/fixture_bench.rs`): every
fixture under `data/fixtures/{lang,modalities}`, warmed up, median of N runs,
single-thread per fixture plus one multi-thread sweep, ids verified equal
before the run counts. Because the released wheel and this build share the
package name, the release is installed into `.release/` (`make bench` does
this) and benched in a subprocess with `PYTHONPATH` pointing there.

One caveat when quoting numbers: the 0.x side is timed on its fastest API
(`encode_batch_fast`), but it still builds `Encoding` objects, while 1.x
returns bare id arrays — part of the speedup is the new API doing strictly
less output work. That is what a user pays end-to-end, but it is not a
model-algorithm-only comparison.

CI runs it in the `python-bindings-bench` job of the Pipeline Benchmark
workflow, posts the table to the run's step summary, and the report job
renders it as a chart (`.github/scripts/render_python_bench.py`) appended to
the benchmark section in the PR description, next to the Rust charts.
