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

# A `Tokenizer` is the contents of a `tokenizer.json`, and can be edited.
tokenizer = Tokenizer.from_file("tokenizer.json")
# Or the `tokenizer.json` of a model repo on the Hub. `huggingface_hub` downloads and caches it,
# with the same cache, login token and offline switch as transformers.
tokenizer = Tokenizer.from_pretrained("openai-community/gpt2", revision="main")

# `compile()` builds a `CompiledTokenizer`, which encodes and decodes. It cannot be changed, so
# threads can share it. Every call builds a new one, and later edits of the `Tokenizer` don't reach it.
compiled = tokenizer.compile()

encoding = compiled.encode("Hello there, how are you?")
encoding.ids             # [101, 7592, 2045, 1010, 2129, 2024, 2017, 1029, 102]
encoding.type_ids        # [0, 0, 0, 0, 0, 0, 0, 0, 0]
encoding.attention_mask  # [1, 1, 1, 1, 1, 1, 1, 1, 1]

# The same fields as numpy arrays. Each read returns a new copy that the caller owns.
encoding.ids_array             # array([101, 7592, 2045, 1010, 2129, 2024, 2017, 1029, 102], dtype=uint32)
encoding.type_ids_array        # array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=uint32)
encoding.attention_mask_array  # array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=uint32)
torch.from_numpy(encoding.ids_array)  # no-copy translation

compiled.encode("Hello", add_special_tokens=False).ids  # no [CLS]/[SEP]
compiled.decode(encoding.ids)                           # "hello there, how are you?"
compiled.decode(encoding.ids_array)                     # a numpy array works too
compiled.decode(encoding.ids, skip_special_tokens=False)

# Pad one call. `length=None` pads each batch to its longest item.
short, long = compiled.encode_batch(
    ["Hello", "Hello there, how are you?"],
    padding=Padding(direction="left", pad_id=0, pad_token="[PAD]"),
)
len(short) == len(long)  # True
short.attention_mask     # [0, 0, 0, 0, 0, 0, 1, 1, 1]
compiled.encode("Hello", padding=Padding(length=16))  # exactly 16 ids

# Or set a default on the `Tokenizer`: every encode of a tokenizer compiled from it pads.
# `padding=None` switches the default off for one call.
tokenizer.padding = Padding(pad_to_multiple_of=8)
padded = tokenizer.compile()
padded.encode("Hello", padding=None)
```

`examples/` holds a runnable script for each of these; `make examples` runs them all.

## Worker processes

Every class here can be pickled, so a tokenizer can be sent to a worker process and its encodings
sent back: `multiprocessing`, a PyTorch `DataLoader`, `datasets.map(num_proc=...)`.

A pickled `CompiledTokenizer` carries its whole `tokenizer.json`, and unpickling compiles it again.
So send the `Tokenizer` once per worker, through the pool's `initializer`, and compile it
there:

```python
def compile_tokenizer(tokenizer):
    global compiled
    compiled = tokenizer.compile()


def encode(text):
    return compiled.encode(text)


with multiprocessing.get_context("spawn").Pool(4, initializer=compile_tokenizer, initargs=(tokenizer,)) as pool:
    encoded = list(pool.imap(encode, texts, chunksize=32))
```

With `fork`, compile before starting the pool instead: the workers inherit the compiled tokenizer
and nothing is pickled. `examples/multiprocessing_workers.py` is the whole script.

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
