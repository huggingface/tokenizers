# tokenizers Python bindings

This module is still experimental, expect the API to change or break.
If used in a production setting, please pin the version number.

## Install

```
uv sync
```


## Usage

```python
from tokenizers import Padding, Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")
# Or the `tokenizer.json` of a model repo on the Hub. `huggingface_hub` downloads and caches it,
# with the same cache, login token and offline switch as transformers.
tokenizer = Tokenizer.from_pretrained("openai-community/gpt2", revision="main")

encoding = tokenizer.encode("Hello there, how are you?")
encoding.ids             # [101, 7592, 2045, 1010, 2129, 2024, 2017, 1029, 102]
encoding.type_ids        # [0, 0, 0, 0, 0, 0, 0, 0, 0]
encoding.attention_mask  # [1, 1, 1, 1, 1, 1, 1, 1, 1]

# The same fields as numpy arrays: read-only views over the encoding, not copies. Reading one
# costs nothing, and an array keeps its encoding alive for as long as the array exists.
encoding.ids_array             # array([101, 7592, 2045, 1010, 2129, 2024, 2017, 1029, 102], dtype=uint32)
encoding.type_ids_array        # array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=uint32)
encoding.attention_mask_array  # array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=uint32)
torch.from_numpy(encoding.ids_array)  # still no copy

tokenizer.encode("Hello", add_special_tokens=False).ids  # no [CLS]/[SEP]
tokenizer.decode(encoding.ids)                           # "hello there, how are you?"
tokenizer.decode(encoding.ids_array)                     # a numpy array works too
tokenizer.decode(encoding.ids, skip_special_tokens=False)

# Override the `Padding` settings. `length=None` pads each batch to its longest
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

`examples/` holds a runnable script for each of these; `make examples` runs them all.

## Worker processes

Every class here can be pickled, so a tokenizer can be sent to a worker process and its encodings
sent back: `multiprocessing`, a PyTorch `DataLoader`, `datasets.map(num_proc=...)`.

```python
def encode(tokenizer, texts):
    return tokenizer.encode_batch(texts)


with multiprocessing.get_context("spawn").Pool(4) as pool:
    encoded = pool.starmap(encode, [(tokenizer, chunk) for chunk in chunks])
```

The tokenizer is pickled again with every task it is sent with, so give each worker one big chunk
of texts rather than many small tasks. `examples/multiprocessing_workers.py` is the whole script.

## Day to day

- `make develop` regenerates the `.pyi` stub, rebuilds the extension (debug) and reinstalls it
  in editable mode.
- `make test` does the above, then runs `tests/` with pytest.
- `make examples` does the above, then runs every script in `examples/`.
- `make stubs` regenerates `python/tokenizers/tokenizers.pyi`, including the members
  `python/tokenizers/__init__.py` attaches to the Rust classes.
- `make style` / `make check-style` regenerate the stub, then format/lint the Rust and
  Python sides and type-check with `ty`. CI runs `make check-style` and fails if the
  regenerated stub differs from the committed one.
