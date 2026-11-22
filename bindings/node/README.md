<p align="center">
  <br>
  <img src="https://huggingface.co/landing/assets/tokenizers/tokenizers-logo.png" width="600"/>
  <br>
<p>
<p align="center">
  <a href="https://badge.fury.io/js/tokenizers">
    <img alt="Build" src="https://badge.fury.io/js/tokenizers.svg">
  </a>
  <a href="https://github.com/huggingface/tokenizers/blob/main/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/tokenizers.svg?color=blue">
  </a>
</p>
<br>

Node.js bindings over the Rust [`tokenizers`](https://github.com/huggingface/tokenizers/tree/main/tokenizers) library.
They load a `tokenizer.json`, encode text into token ids and decode ids back into text. They do not build, edit, save or
train tokenizers.

## Installation

```bash
npm install tokenizers@latest
```

## Usage

```ts
import { Tokenizer } from 'tokenizers'

const tokenizer = Tokenizer.fromFile('tokenizer.json')

const encoding = tokenizer.encode('Who is John?')
console.log(encoding.ids) // Uint32Array of token ids
console.log(encoding.typeIds) // 0 for the first text of a pair, 1 for the second
console.log(encoding.attentionMask) // 1 for each token, 0 for each padding token

console.log(tokenizer.tokenize('Who is John?')) // the token of each id
console.log(tokenizer.decode(encoding.ids)) // back to text

// A pair of texts, and a batch mixing texts and pairs.
const pair = tokenizer.encode(['Who is John?', 'John is a friend.'])
const batch = tokenizer.encodeBatch(['Who is John?', ['Who is John?', 'John is a friend.']], { padding: {} })
```

Every encode method takes the same options: `addSpecialTokens`, `encodeSpecialTokens`, `padding` and `truncation`. The
[Node documentation](https://huggingface.co/docs/tokenizers/node) describes them.

## License

[Apache License 2.0](../../LICENSE)
