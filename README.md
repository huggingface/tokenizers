# Tokenizers

Provides an implementation of today's most used tokenizers with a focus on performances
and versatility. The goal is to make it as easy as possible to construct a Tokenizer, learn a
vocabulary, and then process some text either in real time or in advance.

## What is a Tokenizer

A Tokenizer works as a pipeline taking some raw text as input, going through multiple steps to
finally output a list of `Token`s. The various steps of the pipeline are:
- Some optional `Normalizer`s. An example would be a Unicode normalization step. They take
some raw text as input, and also output raw text `String`.
- An optional `PreTokenizer` which should take some raw text and take care of spliting
as relevant, and pre-processing tokens if needed. Takes a raw text `String` as input, and
outputs a `Vec<String>`.
- A `Model` to do the actual tokenization. An example of `Model` would be `BPE`. Takes
a `Vec<String>` as input, and gives a `Vec<Token>`.
- Some optional `PostProcessor`s. These are in charge of post processing the list of `Token`s
in any relevant way. This includes truncating, adding some padding, ...

## Try the shell

You can try a simple ByteLevel BPE Tokenizer by using the following command. This expects
`vocab.json` and `merges.txt` files, trained with ByteLevel BPE.

```bash
cd tokenizers
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
cargo run --release shell --vocab gpt2-vocab.json --merges gpt2-merges.txt
```
