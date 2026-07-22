<p align="center">
    <br>
    <img src="https://huggingface.co/landing/assets/tokenizers/tokenizers-logo.png" width="600"/>
    <br>
<p>
<p align="center">
    <img alt="Build" src="https://github.com/huggingface/tokenizers/workflows/Rust/badge.svg">
    <a href="https://github.com/huggingface/tokenizers/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/tokenizers.svg?color=blue&cachedrop">
    </a>
    <a href="https://pepy.tech/project/tokenizers">
        <img src="https://pepy.tech/badge/tokenizers/week" />
    </a>
</p>

Provides an implementation of today's most used tokenizers, with a focus on performance and
versatility.

## Main features:

 - Train new vocabularies and tokenize, using today's most used tokenizers.
 - Extremely fast (both training and tokenization), thanks to the Rust implementation. Takes
   less than 20 seconds to tokenize a GB of text on a server's CPU.
 - Easy to use, but also extremely versatile.
 - Designed for research and production.
 - The Rust library additionally tracks alignments (which part of the original
   sentence a token comes from) and does the full pre-processing: truncate,
   pad, add the special tokens your model needs. The 1.x Python bindings
   focus on fast encoding and do not expose these yet — see the
   [breaking changes](bindings/python/README.md#breaking-changes-vs-0x).

## Performances
Performances can vary depending on hardware. The Python bindings ship a
benchmark against the released wheel
([bindings/python/benches/bench_vs_release.py](bindings/python/benches/bench_vs_release.py));
the Rust core has the equivalent fixture benchmark
(`tokenizers/tk-encode/examples/fixture_bench.rs`). Both run in the Pipeline
Benchmark workflow in CI.


## Bindings

We provide bindings to the following languages (more to come!):
  - [Rust](https://github.com/huggingface/tokenizers/tree/main/tokenizers) (Original implementation)
  - [Python](https://github.com/huggingface/tokenizers/tree/main/bindings/python)
  - [Node.js](https://github.com/huggingface/tokenizers/tree/main/bindings/node)
  - [Ruby](https://github.com/ankane/tokenizers-ruby) (Contributed by @ankane, external repo)

## Installation

`pip install tokenizers` installs the released **0.x** version. The example
below uses the **1.x** rewrite of the Python bindings, which is not released
yet — install it from source (needs a Rust toolchain):

```bash
pip install git+https://github.com/huggingface/tokenizers.git#subdirectory=bindings/python
```

## Quick example using Python:

Choose your model between Byte-Pair Encoding, WordPiece or Unigram and instantiate a tokenizer:

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
```

You can customize how pre-tokenization (e.g., splitting into words) is done:

```python
from tokenizers.pre_tokenizers import Whitespace

tokenizer.pre_tokenizer = Whitespace()
```

Then training your tokenizer on a set of files just takes two lines of codes:

```python
from tokenizers.trainers import BpeTrainer

trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train(files=["wiki.train.raw", "wiki.valid.raw", "wiki.test.raw"], trainer=trainer)
```

Once your tokenizer is trained, encode any text with just one line:
```python
encoding = tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(encoding.tokens)
# ["Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?"]
```

`encode` returns an `Encoding` — ids, tokens, and the masks a model consumes.
The emoji comes out as `[UNK]`: it never appeared in the training files, so it
is not in the vocabulary, and BPE falls back to the `unk_token` we configured
above. When you only need the ids, `encode_ids` returns them as a
`numpy.uint32` array with no copy.

More in [bindings/python](bindings/python) — its README and `examples/`
cover loading pretrained tokenizers, threading and async, and training. (The
[hosted documentation](https://huggingface.co/docs/tokenizers/index) still
describes the released 0.x API.)
