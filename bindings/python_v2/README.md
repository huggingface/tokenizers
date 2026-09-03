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
- `Encoding`: `ids`, `type_ids` and `attention_mask`, the fields the pipeline computes. No
  offsets, no overflow, no special-tokens mask. Read-only.
- `Padding`: how to pad, handed to `from_file` or assigned to `Tokenizer.padding`. Without one,
  the file's own `padding` block applies, which for most files means no padding. Read-only.

See `python/tokenizers/__init__.pyi` for the exact signatures; it is regenerated from the Rust
sources by `make stubs`.

## Install

```
pip install -e '.[testing]'
```

That's it: one editable install and `import tokenizers` works, with `pytest`, `ruff` and
`ty` alongside it. `uv pip install -e '.[testing]'` works the same way and is faster.

## Usage

```python
from tokenizers import Padding, Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")

encoding = tokenizer.encode("Hello there, how are you?")
encoding.ids             # [101, 7592, 2045, 1010, 2129, 2024, 2017, 1029, 102]
encoding.type_ids        # [0, 0, 0, 0, 0, 0, 0, 0, 0]
encoding.attention_mask  # [1, 1, 1, 1, 1, 1, 1, 1, 1]

tokenizer.encode("Hello", add_special_tokens=False).ids  # no [CLS]/[SEP]
tokenizer.decode(encoding.ids)                           # "hello there, how are you?"
tokenizer.decode(encoding.ids, skip_special_tokens=False)

# A `Padding` replaces what the file declares. `length=None` pads each batch to its longest
# item; `None` switches padding off.
tokenizer.padding = Padding(direction="left", pad_id=0, pad_token="[PAD]")
short, long = tokenizer.encode_batch(["Hello", "Hello there, how are you?"])
len(short) == len(long)  # True
short.attention_mask     # [0, 0, 0, 0, 0, 0, 1, 1, 1]

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
- `make stubs` regenerates `python/tokenizers/__init__.pyi` from the extension actually built,
  via `pyo3-introspection` (`tools/stub-gen`). It also regenerates
  `python/tokenizers/tokenizers.pyi`, needed only because `__init__.py` imports from a
  same-named compiled submodule (`tokenizers.tokenizers`, per `[tool.maturin]` in
  `pyproject.toml`) that `ty` can't otherwise resolve. Run it after changing a method
  signature, a doc comment, or the set of exported classes, or just call `make style`, which
  does it for you. Never hand-edit the stubs: they're overwritten on the next `make stubs`.
- `make style` / `make check-style` regenerate the stubs, then format/lint the Rust and
  Python sides and type-check with `ty`.
