<p align="center">
    <br>
    <img src="https://huggingface.co/landing/assets/tokenizers/tokenizers-logo.png" width="600"/>
    <br>
<p>
<p align="center">
    <img alt="Build" src="https://github.com/huggingface/tokenizers/workflows/Rust/badge.svg">
    <a href="https://github.com/huggingface/tokenizers/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/tokenizers.svg?color=blue">
    </a>
</p>
<br>

Provides an implementation of today's most used tokenizers, with a focus on performance and
versatility.

## Main features:

 - Train new vocabularies and tokenize, using todays most used tokenizers.
 - Extremely fast (both training and tokenization), thanks to the Rust implementation. Takes
   less than 20 seconds to tokenize a GB of text on a server's CPU.
 - Easy to use, but also extremely versatile.
 - Designed for research and production.
 - Normalization comes with alignments tracking. It's always possible to get the part of the
   original sentence that corresponds to a given token.
 - Does all the pre-processing: Truncate, Pad, add the special tokens your model needs.

<p align="center">
    <br>
    <img src="https://huggingface.co/landing/assets/tokenizers/tokenizers-repo-example.png" />
    <br>
<p>

## Bindings

We provide bindings to the following languages (more to come!):
  - [Rust](https://github.com/huggingface/tokenizers/tree/master/tokenizers) (Original implementation)
  - [Python](https://github.com/huggingface/tokenizers/tree/master/bindings/python)
  - [Node.js](https://github.com/huggingface/tokenizers/tree/master/bindings/node)
