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
 
## Quick examples using Python:

Start using in a matter of seconds:

```python
# Tokenizers provides ultra-fast implementations of most current tokenizers:
>>> from tokenizers import (ByteLevelBPETokenizer,
                            CharBPETokenizer,
                            SentencePieceBPETokenizer,
                            BertWordPieceTokenizer)
# Ultra-fast => they can encode 1GB of text in ~20sec on a standard server's CPU
# Tokenizers can be easily instantiated from standard files
>>> tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)
Tokenizer(vocabulary_size=30522, model=BertWordPiece, add_special_tokens=True, unk_token=[UNK], 
          sep_token=[SEP], cls_token=[CLS], clean_text=True, handle_chinese_chars=True, 
          strip_accents=True, lowercase=True, wordpieces_prefix=##)

# Tokenizers provide exhaustive outputs: tokens, mapping to original string, attention/special token masks.
# They also handle model's max input lengths as well as padding (to directly encode in padded batches)
>>> output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
Encoding(num_tokens=13, attributes=[ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing, original_str, normalized_str])
>>> print(output.ids, output.tokens, output.offsets)
[101, 7592, 1010, 1061, 1005, 2035, 999, 2129, 2024, 2017, 100, 1029, 102]
['[CLS]', 'hello', ',', 'y', "'", 'all', '!', 'how', 'are', 'you', '[UNK]', '?', '[SEP]']
[(0, 0), (0, 5), (5, 6), (7, 8), (8, 9), (9, 12), (12, 13), (14, 17), (18, 21), (22, 25), (26, 27),
 (28, 29), (0, 0)]
# Here is an example using the offsets mapping to retrieve the string corresponding to the 10th token:
>>> output.original_str[output.offsets[10]]
'😁'
```

And training a new vocabulary is just as easy:

```python
# You can also train a BPE/Byte-levelBPE/WordPiece vocabulary on your own files
>>> tokenizer = ByteLevelBPETokenizer()
>>> tokenizer.train(["wiki.test.raw"], vocab_size=20000)
[00:00:00] Tokenize words                 ████████████████████████████████████████   20993/20993
[00:00:00] Count pairs                    ████████████████████████████████████████   20993/20993
[00:00:03] Compute merges                 ████████████████████████████████████████   19375/19375
```
 
## Contributors
  
[![](https://sourcerer.io/fame/clmnt/huggingface/tokenizers/images/0)](https://sourcerer.io/fame/clmnt/huggingface/tokenizers/links/0)[![](https://sourcerer.io/fame/clmnt/huggingface/tokenizers/images/1)](https://sourcerer.io/fame/clmnt/huggingface/tokenizers/links/1)[![](https://sourcerer.io/fame/clmnt/huggingface/tokenizers/images/2)](https://sourcerer.io/fame/clmnt/huggingface/tokenizers/links/2)[![](https://sourcerer.io/fame/clmnt/huggingface/tokenizers/images/3)](https://sourcerer.io/fame/clmnt/huggingface/tokenizers/links/3)[![](https://sourcerer.io/fame/clmnt/huggingface/tokenizers/images/4)](https://sourcerer.io/fame/clmnt/huggingface/tokenizers/links/4)[![](https://sourcerer.io/fame/clmnt/huggingface/tokenizers/images/5)](https://sourcerer.io/fame/clmnt/huggingface/tokenizers/links/5)[![](https://sourcerer.io/fame/clmnt/huggingface/tokenizers/images/6)](https://sourcerer.io/fame/clmnt/huggingface/tokenizers/links/6)[![](https://sourcerer.io/fame/clmnt/huggingface/tokenizers/images/7)](https://sourcerer.io/fame/clmnt/huggingface/tokenizers/links/7)

