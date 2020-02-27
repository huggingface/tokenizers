# [0.5.0](https://github.com/huggingface/tokenizers/compare/node-v0.4.1...node-v0.5.0) (2020-02-26)

### BREAKING CHANGES

- The `Encoding` object now exposes getters instead of `get...` methods (except for `getOriginalString`) ([9179968](https://github.com/huggingface/tokenizers/commit/917996841df2b3385e0212c9d7e9910d4e0d3fbf))
- `BertWordPieceTokenizer` now cleans up some tokenization artifacts by default while decoding ([#145](https://github.com/huggingface/tokenizers/issues/145)) ([#147](https://github.com/huggingface/tokenizers/pull/147))

### Features

- `Encoding` exposes a new `length` property ([9179968](https://github.com/huggingface/tokenizers/commit/917996841df2b3385e0212c9d7e9910d4e0d3fbf))
- Add a new `stripNormalizer` ([#140](https://github.com/huggingface/tokenizers/pull/140)) ([815d743](https://github.com/huggingface/tokenizers/commit/815d743461f9067ab38237862b7be8114d422300))
- `ByteLevelBPETokenizer` and `BPETokenizer` accept more options ([946ac1a](https://github.com/huggingface/tokenizers/commit/946ac1a9517c3090064e9a972ad71a5cf25b7e7f))
- Add `save` method to `Model` class ([aebc97e](https://github.com/huggingface/tokenizers/commit/aebc97eaf34260c9ed7689dd5e087bf8c8af59fc))
- Improved padding performances ([b30be3b](https://github.com/huggingface/tokenizers/commit/b30be3b2bda977b65f9bdb384258829b2bd91e3d)) ([0dc857e](https://github.com/huggingface/tokenizers/commit/0dc857ea8c557532a52628a6bc80141e65e6d974))

### Fixes

- Methods accepting optional arguments now handle explicit `undefined` correctly ([0fe22a7](https://github.com/huggingface/tokenizers/commit/0fe22a7c1c23f8d992f502a3a582e5212b8281ac))
- Special tokens are now declared only if present in the vocabulary ([b70283c](https://github.com/huggingface/tokenizers/commit/b70283c3050056958e8ba020b0386451cc6df80c))
- Add missing mask/padding special tokens in wordpiece tokenizer ([b70283c](https://github.com/huggingface/tokenizers/commit/b70283c3050056958e8ba020b0386451cc6df80c))

## [0.4.1](https://github.com/huggingface/tokenizers/compare/node-v0.4.0...node-v0.4.1) (2020-02-11)

### Bug Fixes

- Fix punctuation in BertWordPieceTokenizer (Thanks to @Mansterteddy with [#134](https://github.com/huggingface/tokenizers/pull/134))

# [0.4.0](https://github.com/huggingface/tokenizers/compare/node-v0.3.1...node-v0.4.0) (2020-02-05)

### BREAKING CHANGES

- `getOverflowing()` method on `Encoding` now returns all the overflowing `Encoding`s at once ([#77](https://github.com/huggingface/tokenizers/pull/77)) ([0094393](https://github.com/huggingface/tokenizers/commit/0094393610623bafc269790cd1be81fd1474583a))

### Features

- Add `setTruncation`, `disableTruncation`, `setPadding` and `disablePadding` methods in `Tokenizer` and `BaseTokenizer` ([#109](https://github.com/huggingface/tokenizers/pull/109)) ([78e2690](https://github.com/huggingface/tokenizers/commit/78e26905a735e14e67590cb09ddb42ed141c455b))
- Expose tokenizer / truncation / padding configuration in `BaseTokenizer` ([#126](https://github.com/huggingface/tokenizers/pull/126)) ([cb8585b](https://github.com/huggingface/tokenizers/commit/cb8585bc4eb8037c52049da677e4791857231f03))
- Expose `addTokens`, `addSpecialTokens`, `idToToken` and `tokenToId` in `BaseTokenizer` ([7051480](https://github.com/huggingface/tokenizers/commit/7051480c333f88bef80aa6846b66032a2d47383c))
- Add `getOriginalString()` method on `Encoding` ([a14c633](https://github.com/huggingface/tokenizers/commit/a14c63343b217a2c501359bec52baf717e3a05ef))
- Add `charDelimiterSplitPreTokenizer`: a new `PreTokenizer` that allows splitting sequences on the given delimiter (works like `.split(delimiter)`) ([#114](https://github.com/huggingface/tokenizers/pull/114)) ([6165910](https://github.com/huggingface/tokenizers/commit/6165910ca66b6bfd9fd996aa38c4c0b2b6505953))
- Add `robertaProcessing` as a new `PostProcessor` ([#111](https://github.com/huggingface/tokenizers/pull/111)) ([6524f09](https://github.com/huggingface/tokenizers/commit/6524f09e991c3a52c839d8eb01bfa41e81fde1d1))

### Bug Fixes

- Correctly truncate with `OnlyFirst` and `OnlySecond` strategies ([#108](https://github.com/huggingface/tokenizers/issues/108)) ([6d532fe](https://github.com/huggingface/tokenizers/commit/6d532fedb1d3626328828304a5c39807733d2fa1))
- Fix default special tokens in `BertWordPieceTokenizer` ([10e2d28](https://github.com/huggingface/tokenizers/commit/10e2d286caf517f0977c04cf8e1924aed90403c9))
- Fix return type of `getSpecialTokensMask` on `Encoding` ([9770be5](https://github.com/huggingface/tokenizers/commit/9770be566175dc9c44dd7dcaa00a57d0e4ca632b))
- Actually add special tokens in tokenizers implementations ([acef252](https://github.com/huggingface/tokenizers/commit/acef252dacc43adc414175cfc325668ad1488753))
