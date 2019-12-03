[![PyPI version](https://badge.fury.io/py/tokenizers.svg)](https://badge.fury.io/py/tokenizers)

# Tokenizers

A fast and easy to use implementation of today's most used tokenizers.

 - High Level design: [master](https://github.com/huggingface/tokenizers)

### Installation

#### With pip:
```
pip install tokenizers
```

#### From sources:

To use this method, you need to have the Rust nightly toolchain installed.
```
# Install with:
curl https://sh.rustup.rs -sSf | sh -s -- -default-toolchain nightly-2019-11-01 -
export PATH="$HOME/.cargo/bin:$PATH"

# Or select the right toolchain:
rustup default nightly-2019-11-01
```

Once Rust is installed and using the right toolchain you can do the following.
```
git clone https://github.com/huggingface/tokenizers
cd tokenizers/bindings/python

# Create a virtual env (you can use yours as well)
python -m venv .env
source .env/bin/activate

# Install `tokenizers` in the current virtual env
pip install maturin
maturin develop --release
```

### Usage

```python
from tokenizers import Tokenizer, models, pre_tokenizers, decoders

# Load a BPE Model
vocab = "./path/to/vocab.json"
merges = "./path/to/merges.txt"
bpe = models.BPE.from_files(vocab, merges)

# Initialize a tokenizer
tokenizer = Tokenizer(bpe)

# Customize pre-tokenization and decoding
tokenizer.with_pre_tokenizer(pre_tokenizers.ByteLevel.new())
tokenizer.with_decoder(decoders.ByteLevel.new())

# And then tokenize:
encoded = tokenizer.encode("I can feel the magic, can you?")
print(encoded)

# Or tokenize multiple sentences at once:
encoded = tokenizer.encode_batch([
	"I can feel the magic, can you?",
	"The quick brown fox jumps over the lazy dog"
])
print(encoded)
```
