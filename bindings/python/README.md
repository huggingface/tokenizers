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

We provide some pre-build tokenizers to cover the most common cases. You can easily load one of
these using some `vocab.json` and `merges.txt` files:

```python
from tokenizers import CharBPETokenizer

# Initialize a tokenizer
vocab = "./path/to/vocab.json"
merges = "./path/to/merges.txt"
tokenizer = CharBPETokenizer(vocab, merges)

# And then encode:
encoded = tokenizer.encode("I can feel the magic, can you?")
print(encoded.ids)
print(encoded.tokens)
```

And you can train them just as simply:

```python
from tokenizers import CharBPETokenizer

# Initialize a tokenizer
tokenizer = CharBPETokenizer()

# Then train it!
tokenizer.train([ "./path/to/files/1.txt", "./path/to/files/2.txt" ])

# Now, let's use it:
encoded = tokenizer.encode("I can feel the magic, can you?")

# And finally save it somewhere
tokenizer.save("./path/to/directory/my-bpe.tokenizer.json")
```

#### Provided Tokenizers

 - `CharBPETokenizer`: The original BPE
 - `ByteLevelBPETokenizer`: The byte level version of the BPE
 - `SentencePieceBPETokenizer`: A BPE implementation compatible with the one used by SentencePiece
 - `BertWordPieceTokenizer`: The famous Bert tokenizer, using WordPiece

All of these can be used and trained as explained above!

### Build your own

Whenever these provided tokenizers don't give you enough freedom, you can build your own tokenizer,
by putting all the different parts you need together.
You can check how we implemented the [provided tokenizers](https://github.com/huggingface/tokenizers/tree/master/bindings/python/py_src/tokenizers/implementations) and adapt them easily to your own needs.

#### Building a byte-level BPE

Here is an example showing how to build your own byte-level BPE by putting all the different pieces
together, and then saving it to a single file:

```python
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

# Initialize a tokenizer
tokenizer = Tokenizer(models.BPE())

# Customize pre-tokenization and decoding
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

# And then train
trainer = trainers.BpeTrainer(vocab_size=20000, min_frequency=2)
tokenizer.train([
	"./path/to/dataset/1.txt",
	"./path/to/dataset/2.txt",
	"./path/to/dataset/3.txt"
], trainer=trainer)

# And Save it
tokenizer.save("byte-level-bpe.tokenizer.json", pretty=True)
```

Now, when you want to use this tokenizer, this is as simple as:

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("byte-level-bpe.tokenizer.json")

encoded = tokenizer.encode("I can feel the magic, can you?")
```
