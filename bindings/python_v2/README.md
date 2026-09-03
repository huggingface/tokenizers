# tokenizers (Python bindings, pipeline encode path)

Lives in `bindings/python_v2`, next to the existing `bindings/python`, while the two coexist.
`bindings/python` wraps the pre-v1 engine, which has been removed; that crate no longer builds.
This is a from-scratch restart against the pipeline encode path (`tk-encode`), the way
`bindings/node` was already rebuilt. It imports as `tokenizers`, the same name `bindings/python`
publishes under, so don't install both into the same environment.

The surface is three classes:

- `Tokenizer`: `from_file` reads a `tokenizer.json`, `encode`/`encode_batch` turn text into
  `Encoding`s, `decode` turns ids back into text. `padding` is the only attribute you can
  change on one.
- `Encoding`: `ids`, `type_ids` and `attention_mask`, the fields the pipeline computes, as
  read-only numpy arrays over the encoding's own memory. No offsets, no overflow, no
  special-tokens mask.
- `Padding`: how to pad, handed to `from_file` or assigned to `Tokenizer.padding`. Without one,
  the file's own `padding` block applies, which for most files means no padding. Read-only.

See `python/tokenizers/tokenizers.pyi` for the exact signatures; it is regenerated from the Rust
sources by `make stubs`.

## Install

```
uv sync
```

One command: it builds the extension, installs it in editable mode into `.venv`, and installs
the `dev` dependency group (`maturin`, `pytest`, `ruff`, `ty`). Without uv,
`pip install -e . --group dev` does the same with pip 25.1 or newer. `numpy` is the one
runtime dependency.

## Usage

```python
from tokenizers import Padding, Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")

encoding = tokenizer.encode("Hello there, how are you?")
encoding.ids             # array([101, 7592, 2045, 1010, 2129, 2024, 2017, 1029, 102], dtype=uint32)
encoding.type_ids        # array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=uint32)
encoding.attention_mask  # array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=uint32)
encoding.ids.tolist()    # [101, 7592, 2045, 1010, 2129, 2024, 2017, 1029, 102]

# The arrays are read-only views over the encoding, not copies. Reading a field costs nothing,
# and an array keeps its encoding alive for as long as the array exists.

tokenizer.encode("Hello", add_special_tokens=False).ids  # no [CLS]/[SEP]
tokenizer.decode(encoding.ids)                           # "hello there, how are you?"
tokenizer.decode([101, 7592, 102])                       # any sequence of ints works too
tokenizer.decode(encoding.ids, skip_special_tokens=False)

# A `Padding` replaces what the file declares. `length=None` pads each batch to its longest
# item; `None` switches padding off.
tokenizer.padding = Padding(direction="left", pad_id=0, pad_token="[PAD]")
short, long = tokenizer.encode_batch(["Hello", "Hello there, how are you?"])
len(short) == len(long)  # True
short.attention_mask     # array([0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=uint32)

tokenizer.padding = Padding(length=16)            # every encoding is exactly 16 ids
tokenizer.padding = Padding(pad_to_multiple_of=8)
tokenizer.padding = None

# Or fix it at construction.
Tokenizer.from_file("tokenizer.json", padding=Padding(length=16))
```

`examples/` has the same as runnable scripts against the fixtures `make test` fetches:
`make examples` runs them.

## Day to day

- `make develop` rebuilds the extension (debug, fast) and reinstalls it in editable mode.
  This is the inner loop; it does not touch the `.pyi` stubs.
- `make test` does the above, then runs `tests/` with pytest.
- `make examples` does the above, then runs every script in `examples/`.
- `make stubs` regenerates `python/tokenizers/tokenizers.pyi`, the stub for the compiled
  module, from a release build of it (`tools/stub-gen`, over `pyo3-introspection`).
  `__init__.py` re-exports everything from that module, so one stub covers the package. Run it
  after changing a method signature, a doc comment, or the set of exported classes, or just
  call `make style`, which does it for you. Never hand-edit the stub: it's overwritten on the
  next `make stubs`.
- `make style` / `make check-style` regenerate the stubs, then format/lint the Rust and
  Python sides and type-check with `ty`.
