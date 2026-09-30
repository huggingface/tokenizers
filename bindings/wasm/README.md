<p align="center">
  <br>
  <img src="https://huggingface.co/landing/assets/tokenizers/tokenizers-logo.png" width="600"/>
  <br>
<p>
<p align="center">
  <a href="https://badge.fury.io/js/tokenizers-web">
    <img alt="Build" src="https://badge.fury.io/js/tokenizers-web.svg">
  </a>
  <a href="https://github.com/huggingface/tokenizers/blob/main/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/tokenizers.svg?color=blue">
  </a>
</p>
<br>

WebAssembly bindings over the [Rust](https://github.com/huggingface/tokenizers/tree/main/tokenizers) `tokenizers` library,
for fast tokenization in the Browser.

## Requirements

This package requires somewhat recent features of WebAssembly, including SIMD instructions.

| Browser                | Minimum version |
| ---------------------- | --------------- |
| Chrome                 | 96              |
| Edge                   | 96              |
| Firefox                | 89              |
| Safari (macOS and iOS) | 16.4            |

In older browsers, importing the package throws a `WebAssembly.CompileError`.

## Installation

```bash
npm install tokenizers-web@latest
```

## Basic example

You'll find below a basic example of how to use this package.

You can also find in [the examples section](https://github.com/huggingface/tokenizers/tree/main/bindings/wasm/examples)
of the repository a simple application showcasing how to use `tokenizers-web` with React.

```typescript
import { Tokenizer } from "tokenizers-web";

const tokenizer = await Tokenizer.from_pretrained("openai-community/gpt2");
const ids = tokenizer.encode("Hello world!");
console.log(ids); // Uint32Array [15496, 995, 0]
console.log(tokenizer.decode_tokens(ids)); // ["Hello", "Ġworld", "!"]
console.log(tokenizer.decode(ids)); // "Hello world!"
```

The package loads its WebAssembly module on import, and the garbage collector frees a tokenizer once it is unreachable.
To release its memory right away, call `tokenizer.free()` or declare it with `using`.
