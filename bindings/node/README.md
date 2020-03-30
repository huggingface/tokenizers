<p align="center">
  <br>
  <img src="https://huggingface.co/landing/assets/tokenizers/tokenizers-logo.png" width="600"/>
  <br>
<p>
<p align="center">
  <a href="https://badge.fury.io/js/tokenizers">
    <img alt="Build" src="https://badge.fury.io/js/tokenizers.svg">
  </a>
  <a href="https://github.com/huggingface/tokenizers/blob/master/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/tokenizers.svg?color=blue">
  </a>
</p>
<br>

NodeJS implementation of today's most used tokenizers, with a focus on performance and
versatility. Bindings over the [Rust](https://github.com/huggingface/tokenizers/tree/master/tokenizers) implementation.
If you are interested in the High-level design, you can go check it there.

## Main features

 - Train new vocabularies and tokenize using 4 pre-made tokenizers (Bert WordPiece and the 3
   most common BPE versions).
 - Extremely fast (both training and tokenization), thanks to the Rust implementation. Takes
   less than 20 seconds to tokenize a GB of text on a server's CPU.
 - Easy to use, but also extremely versatile.
 - Designed for research and production.
 - Normalization comes with alignments tracking. It's always possible to get the part of the
   original sentence that corresponds to a given token.
 - Does all the pre-processing: Truncate, Pad, add the special tokens your model needs.

## Installation

```bash
npm install tokenizers@latest
```

## Basic example

```ts
import { BertWordPieceTokenizer } from "tokenizers";

const wordPieceTokenizer = await BertWordPieceTokenizer.fromOptions({ vocabFile: "./vocab.txt" });
const wpEncoded = await wordPieceTokenizer.encode("Who is John?", "John is a teacher");

console.log(wpEncoded.length);
console.log(wpEncoded.tokens);
console.log(wpEncoded.ids);
console.log(wpEncoded.attentionMask);
console.log(wpEncoded.offsets);
console.log(wpEncoded.overflowing);
console.log(wpEncoded.specialTokensMask);
console.log(wpEncoded.typeIds);
console.log(wpEncoded.wordIndexes);
```

## Provided Tokenizers

 - `BPETokenizer`: The original BPE
 - `ByteLevelBPETokenizer`: The byte level version of the BPE
 - `SentencePieceBPETokenizer`: A BPE implementation compatible with the one used by SentencePiece
 - `BertWordPieceTokenizer`: The famous Bert tokenizer, using WordPiece

## License

[Apache License 2.0](../../LICENSE)
