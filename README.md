<p align="center">
    <br>
    <img src="https://huggingface.co/landing/assets/tokenizers/tokenizers-logo.png" width="600"/>
    <br>
<p>
<p align="center">
    <img alt="Build" src="https://github.com/huggingface/tokenizers/workflows/Rust/badge.svg">
    <a href="https://github.com/huggingface/tokenizers/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/tokenizers.svg?color=blue&cachedrop">
    </a>
    </a>
    <a href="https://pepy.tech/project/tokenizers/week">
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
 - Normalization comes with alignments tracking. It's always possible to get the part of the
   original sentence that corresponds to a given token.
 - Does all the pre-processing: Truncate, Pad, add the special tokens your model needs.

## Bindings

We provide bindings to the following languages (more to come!):
  - [Rust](https://github.com/huggingface/tokenizers/tree/master/tokenizers) (Original implementation)
  - [Python](https://github.com/huggingface/tokenizers/tree/master/bindings/python)
  - [Node.js](https://github.com/huggingface/tokenizers/tree/master/bindings/node)
 
## Quick example using Python:

Choose your model between Byte-Pair Encoding, WordPiece or Unigram and instantiate a tokenizer:

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer(BPE())
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
tokenizer.train(trainer, ["wiki.train.raw", "wiki.valid.raw", "wiki.test.raw"])
```

Once your tokenizer is trained, encode any text with just one line:
```python
output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(output.tokens)
# ["Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?"]
```

Check the [python documentation](https://huggingface.co/docs/tokenizers/python/latest) or the 
[python quicktour](https://huggingface.co/docs/tokenizers/python/latest/quicktour.html) to learn more!
 
## Contributors
  
[![](https://sourcerer.io/fame/clmnt/huggingface/tokenizers/images/0)](https://sourcerer.io/fame/clmnt/huggingface/tokenizers/links/0)[![](https://sourcerer.io/fame/clmnt/huggingface/tokenizers/images/1)](https://sourcerer.io/fame/clmnt/huggingface/tokenizers/links/1)[![](https://sourcerer.io/fame/clmnt/huggingface/tokenizers/images/2)](https://sourcerer.io/fame/clmnt/huggingface/tokenizers/links/2)[![](https://sourcerer.io/fame/clmnt/huggingface/tokenizers/images/3)](https://sourcerer.io/fame/clmnt/huggingface/tokenizers/links/3)[![](https://sourcerer.io/fame/clmnt/huggingface/tokenizers/images/4)](https://sourcerer.io/fame/clmnt/huggingface/tokenizers/links/4)[![](https://sourcerer.io/fame/clmnt/huggingface/tokenizers/images/5)](https://sourcerer.io/fame/clmnt/huggingface/tokenizers/links/5)[![](https://sourcerer.io/fame/clmnt/huggingface/tokenizers/images/6)](https://sourcerer.io/fame/clmnt/huggingface/tokenizers/links/6)[![](https://sourcerer.io/fame/clmnt/huggingface/tokenizers/images/7)](https://sourcerer.io/fame/clmnt/huggingface/tokenizers/links/7)

