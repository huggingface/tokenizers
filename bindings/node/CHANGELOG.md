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
