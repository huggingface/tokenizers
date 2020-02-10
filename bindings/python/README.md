<p align="center">
    <br>
    <img src="https://huggingface.co/landing/assets/tokenizers/tokenizers-logo.png" width="600"/>
    <br>
<p>
<p align="center">
    <a href="https://badge.fury.io/py/tokenizers">
         <img alt="Build" src="https://badge.fury.io/py/tokenizers.svg">
    </a>
    <a href="https://github.com/huggingface/tokenizers/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/tokenizers.svg?color=blue">
    </a>
</p>
<br>

# Tokenizers

Provides an implementation of today's most used tokenizers, with a focus on performance and
versatility.

Bindings over the [Rust](https://github.com/huggingface/tokenizers/tree/master/tokenizers) implementation.
If you are interested in the High-level design, you can go check it there.

Otherwise, let's dive in!

## Main features:

 - Train new vocabularies and tokenize using 4 pre-made tokenizers (Bert WordPiece and the 3
   most common BPE versions).
 - Extremely fast (both training and tokenization), thanks to the Rust implementation. Takes
   less than 20 seconds to tokenize a GB of text on a server's CPU.
 - Easy to use, but also extremely versatile.
 - Designed for research and production.
 - Normalization comes with alignments tracking. It's always possible to get the part of the
   original sentence that corresponds to a given token.
 - Does all the pre-processing: Truncate, Pad, add the special tokens your model needs.

### Installation

#### With pip:

```bash
pip install tokenizers
```

#### From sources:

To use this method, you need to have the Rust installed:

```bash
# Install with:
curl https://sh.rustup.rs -sSf | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"
```

Once Rust is installed, you can compile doing the following

```bash
git clone https://github.com/huggingface/tokenizers
cd tokenizers/bindings/python

# Create a virtual env (you can use yours as well)
python -m venv .env
source .env/bin/activate

# Install `tokenizers` in the current virtual env
pip install setuptools_rust
python setup.py install
```

### Using the provided Tokenizers

Using a pre-trained tokenizer is really simple:

```python
from tokenizers import BPETokenizer

# Initialize a tokenizer
vocab = "./path/to/vocab.json"
merges = "./path/to/merges.txt"
tokenizer = BPETokenizer(vocab, merges)

# And then encode:
encoded = tokenizer.encode("I can feel the magic, can you?")
print(encoded.ids)
print(encoded.tokens)
```

And you can train yours just as simply:

```python
from tokenizers import BPETokenizer

# Initialize a tokenizer
tokenizer = BPETokenizer()

# Then train it!
tokenizer.train([ "./path/to/files/1.txt", "./path/to/files/2.txt" ])

# And you can use it
encoded = tokenizer.encode("I can feel the magic, can you?")

# And finally save it somewhere
tokenizer.save("./path/to/directory", "my-bpe")
```

### Provided Tokenizers

 - `BPETokenizer`: The original BPE
 - `ByteLevelBPETokenizer`: The byte level version of the BPE
 - `SentencePieceBPETokenizer`: A BPE implementation compatible with the one used by SentencePiece
 - `BertWordPieceTokenizer`: The famous Bert tokenizer, using WordPiece

All of these can be used and trained as explained above!

### Build your own

You can also easily build your own tokenizers, by putting all the different parts
you need together:

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
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()

# And then encode:
encoded = tokenizer.encode("I can feel the magic, can you?")
print(encoded.ids)
print(encoded.tokens)

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
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()

# And then train
trainer = trainers.BpeTrainer(vocab_size=20000, min_frequency=2)
tokenizer.train(trainer, [
	"./path/to/dataset/1.txt",
	"./path/to/dataset/2.txt",
	"./path/to/dataset/3.txt"
])

# Now we can encode
encoded = tokenizer.encode("I can feel the magic, can you?")
print(encoded)
```
