[![PyPI version](https://badge.fury.io/py/tokenizers.svg)](https://badge.fury.io/py/tokenizers)

# Tokenizers

A fast and easy to use implementation of today's most used tokenizers.

 - High Level design: [master](https://github.com/huggingface/tokenizers)

This API is currently in the process of being stabilized. We might introduce breaking changes
really often in the coming days/weeks, so use at your own risks.

### Installation

#### With pip:

```bash
pip install tokenizers
```

#### From sources:

To use this method, you need to have the Rust nightly toolchain installed.

```bash
# Install with:
curl https://sh.rustup.rs -sSf | sh -s -- -default-toolchain nightly-2019-11-01 -y
export PATH="$HOME/.cargo/bin:$PATH"

# Or select the right toolchain:
rustup default nightly-2019-11-01
```

Once Rust is installed and using the right toolchain you can do the following.

```bash
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

#### Use a pre-trained tokenizer

```python
from tokenizers import Tokenizer, models, pre_tokenizers, decoders

# Load a BPE Model
vocab = "./path/to/vocab.json"
merges = "./path/to/merges.txt"
bpe = models.BPE.from_files(vocab, merges)

# Initialize a tokenizer
tokenizer = Tokenizer(bpe)

# Customize pre-tokenization and decoding
tokenizer.with_pre_tokenizer(pre_tokenizers.ByteLevel.new(True))
tokenizer.with_decoder(decoders.ByteLevel.new())

# And then encode:
encoded = tokenizer.encode("I can feel the magic, can you?")
print(encoded)

# Or tokenize multiple sentences at once:
encoded = tokenizer.encode_batch([
	"I can feel the magic, can you?",
	"The quick brown fox jumps over the lazy dog"
])
print(encoded)
```

#### Train a new tokenizer

```python
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

# Initialize a tokenizer
tokenizer = Tokenizer(models.BPE.empty())

# Customize pre-tokenization and decoding
tokenizer.with_pre_tokenizer(pre_tokenizers.ByteLevel.new(True))
tokenizer.with_decoder(decoders.ByteLevel.new())

# And then train
trainer = trainers.BpeTrainer.new(vocab_size=20000, min_frequency=2)
tokenizer.train(trainer, [
	"./path/to/dataset/1.txt",
	"./path/to/dataset/2.txt",
	"./path/to/dataset/3.txt"
])

# Now we can encode
encoded = tokenizer.encode("I can feel the magic, can you?")
print(encoded)
```
