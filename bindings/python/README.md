# tokenizers Python bindings

This module is still experimental, expect the API to change or break.
If used in a production setting, please pin the version number.

## Install

```
pip install --pre tokenizers
```


## Usage

```python
from tokenizers import Padding, Tokenizer

# A `Tokenizer` represents a tokenizer's spec (json) and can be edited.
# You can load it from the JSON file
tokenizer = Tokenizer.from_file("tokenizer.json")
# Or load the `tokenizer.json` of a model repo on the Hub.
# Downloads go through the `huggingface_hub` package
tokenizer = Tokenizer.from_pretrained("openai-community/gpt2", revision="main")

# `compile()` compiles the specification to an optimized tokenizer in rust.
# The resulting `CompiledTokenizer` object can encode text to token ids and decode ids back to text.
# It's immutable and can be shared among threads.
compiled = tokenizer.compile()

encoding = compiled.encode("Hello there, how are you?")
encoding.ids             # [101, 7592, 2045, 1010, 2129, 2024, 2017, 1029, 102]
encoding.type_ids        # [0, 0, 0, 0, 0, 0, 0, 0, 0]
encoding.attention_mask  # [1, 1, 1, 1, 1, 1, 1, 1, 1]

# The same fields for array libraries, through DLPack. Each read is an `int64` copy.
torch.tensor(encoding.ids_dlpack)   # tensor([101, 7592, ...]), torch 2.6 or later
np.from_dlpack(encoding.attention_mask_dlpack)  # array([1, 1, ...])

compiled.encode("Hello", add_special_tokens=False).ids  # no [CLS]/[SEP]
compiled.decode(encoding.ids)                           # "hello there, how are you?"
compiled.decode(encoding.ids, skip_special_tokens=False)

# Pad one call. `length=None` pads each batch to its longest item.
short, long = compiled.encode_batch(
    ["Hello", "Hello there, how are you?"],
    padding=Padding(direction="left", pad_id=0, pad_token="[PAD]"),
)
len(short) == len(long)  # True
short.attention_mask     # [0, 0, 0, 0, 0, 0, 1, 1, 1]
compiled.encode("Hello", padding=Padding(length=16))  # exactly 16 ids

# Padding and truncation can also be set on the spec.
# `padding=None` switches the default off for one call.
tokenizer.padding = Padding(pad_to_multiple_of=8)
padded = tokenizer.compile()
padded.encode("Hello", padding=None)
```

`examples/` holds a runnable script for each of these; `make examples` runs them all.

## multiprocessing and Worker processes

Every class can be pickled, so a tokenizer can be sent to a worker process:
`multiprocessing`, a PyTorch `DataLoader`, `datasets.map(num_proc=...)`.

A pickled `CompiledTokenizer` carries the bytes of `tokenizer.json`, and unpickling compiles it again.
Compiling a large vocabulary can take up to a few seconds, so the method for sending the tokenizer
to the workers matters.

- With `"spawn"` or `"forkserver"`, pass the `Tokenizer` (uncompiled spec) to the pool's `initializer` and compile once per worker.
- With `"fork"`, compile before starting the pool. The workers inherit the compiled tokenizer.
- **Never** pass a tokenizer as a task argument, as in `pool.imap(partial(encode, compiled), texts)`.
  The pool would pickle it with every chunk of tasks, and every chunk would compile it again.

Wall time in seconds to encode 50,000 English texts with gemma-3 (262k tokens in its vocabulary),
with 4 workers and `chunksize=64`, on Linux with Python 3.12 and a 16-core AMD EPYC 7R13 (32 threads):

| How the tokenizer reaches the workers | fork | forkserver | spawn |
|---|---|---|---|
| `CompiledTokenizer` compiled before the pool | 2.4s | n/a | n/a |
| `Tokenizer` in `initargs`, compiled per worker | 2.8s | 3.3s | 3.1s |
| `CompiledTokenizer` in `initargs` | 2.4s | 3.9s | 4.3s |
| `CompiledTokenizer` in every task | over 150s | over 150s | over 150s |

For reference, a plain loop in a single process takes 4.9 s.

```python
def compile_tokenizer(tokenizer):
    global compiled
    compiled = tokenizer.compile()


def encode(text):
    return compiled.encode(text)


with multiprocessing.get_context("spawn").Pool(4, initializer=compile_tokenizer, initargs=(tokenizer,)) as pool:
    encoded = list(pool.imap(encode, texts, chunksize=32))
```

## Developers

- `make develop` regenerates the `.pyi` stub, rebuilds the extension (debug) and reinstalls it
  in editable mode.
- `make test` does the above, then runs `tests/` with pytest.
- `make examples` does the above, then runs every script in `examples/`.
- `make stubs` regenerates `python/tokenizers/tokenizers.pyi`, including the members
  `python/tokenizers/__init__.py` attaches to the Rust classes.
- `make style` / `make check-style` regenerate the stub, then format/lint the Rust and
  Python sides and type-check with `ty`. CI runs `make check-style` and fails if the
  regenerated stub differs from the committed one.
