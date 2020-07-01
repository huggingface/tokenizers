# [0.7.0](https://github.com/huggingface/tokenizers/compare/node-v0.6.2...node-v0.7.0) (2020-07-01)

### BREAKING CHANGES

- `robertaProcessing` now handles trimming the offsets (activated by default) ([#236](https://github.com/huggingface/tokenizers/pull/236))
- `charToTokenOffsets`, `charToWordOffsets` and `tokenToWordOffsets` helper functions on `Encoding` instances are removed and replaced by new `wordToTokens`, `wordToChars`, `tokenToChars`, `tokenToWord` and `charToWord` methods ([#234](https://github.com/huggingface/tokenizers/pull/234))
- `encode` and `encodeBatch` methods on a tokenizer now handle pre-tokenized inputs and have their signatures changed ([#249](https://github.com/huggingface/tokenizers/pull/249)). In addition:
  - `encodeTokenized`, `encodeTokenizedBatch` methods are therefore removed
  - `InputSequence`, `EncodeInput` and `EncodeOptions` types are added
- Improve management of the additional vocabulary ([#309](https://github.com/huggingface/tokenizers/pull/309)):
  - New parameter `normalized` in `AddedToken` options, controlling whether a token should be extracted from the normalized version of the input text
  - The `AddedToken` constructor now takes a `special` boolean as second parameter to indicate if the token is special (in this case it won't be normalized)

### Features

- Serialization of a `Tokenizer` and all its parts (`PreTokenizer`, `Normalizer`, ...). This adds some methods to easily save/load an entire tokenizer: new static methods `fromString` / `fromFile`, and instance methods `save` / `toString` on `BaseTokenizer` ([#272](https://github.com/huggingface/tokenizers/pull/272))
- New `padToMultipleOf` parameter for `PaddingOptions`, to pad to a multiple of a specified value ([#289](https://github.com/huggingface/tokenizers/pull/289))
- Improved errors generated during truncation when the provided max length is too low ([02cc977](https://github.com/huggingface/tokenizers/commit/02cc97756ffb9193b5d6d8dfcdeb7bf08adf2516))
- Improve BPE training speeds, by reading files sequentially, but parallelizing the processing of each file ([#276](https://github.com/huggingface/tokenizers/pull/276))
- Use `onig` for byte-level pre-tokenization to remove all the differences with the original implementation from GPT-2 ([#280](https://github.com/huggingface/tokenizers/pull/280))

### Fixes

- Fix various crash when training a BPE model ([#286](https://github.com/huggingface/tokenizers/pull/286))
- Fix a few bugs related to additional vocabulary/tokens ([#309](https://github.com/huggingface/tokenizers/pull/309))

## [0.6.2](https://github.com/huggingface/tokenizers/compare/node-v0.6.1...node-v0.6.2) (2020-04-13)

### Features

- More symbols exposed: `Token`, `BaseTokenizer`, `PaddingConfiguration`, `TruncationConfiguration` ([38d53a7](https://github.com/huggingface/tokenizers/commit/38d53a7b84b2ee86b262eee2de6121351fe03889))
- Expose `setPostProcessor` in `BaseTokenizer` ([38d53a7](https://github.com/huggingface/tokenizers/commit/38d53a7b84b2ee86b262eee2de6121351fe03889))

### Fixes

- Fix the word indexes when there are special tokens ([#226](https://github.com/huggingface/tokenizers/pull/226))
- Fix encoding overflowing offsets ([695ab83](https://github.com/huggingface/tokenizers/commit/695ab8388f5f1a7d63d8aaab9b3762312e0d5ac3))
- Fix Roberta overflowings ([c4ecc6f](https://github.com/huggingface/tokenizers/commit/c4ecc6f7ce7af40c558401a3ec9500732a17f9da))

## [0.6.1](https://github.com/huggingface/tokenizers/compare/node-v0.6.0...node-v0.6.1) (2020-04-01)

### Fixes

- Fix special tokens with wrong id ([b770f36](https://github.com/huggingface/tokenizers/commit/b770f364280af33efeffea8f0003102cda8cf1b7))
- Fix `AddedToken`'s `leftStrip` and `rightStrip` params (thanks @thirdwing) ([85488dd](https://github.com/huggingface/tokenizers/commit/85488dd6330ec7fa64aeb78c1a86b221f77c5ebb))

# [0.6.0](https://github.com/huggingface/tokenizers/compare/node-v0.5.0...node-v0.6.0) (2020-03-30)

### BREAKING CHANGES

- The `getOriginalString` method on `Encoding`s has been removed: this brings a reduction of 70% of the memory footprint. You can use the provided new `slice` function as a replacement to get a subpart of a string according to specified indexes while respecting unicode characters. ([#197](https://github.com/huggingface/tokenizers/pull/197))
- The offsets provided on `Encoding` are now relative to the original string, and not the normalized one anymore ([#197](https://github.com/huggingface/tokenizers/pull/197))
- The added tokens given to `addTokens`, `addSpecialTokens` or `train` methods of a tokenizer can now be instances of `AddedToken` to provide more control over these tokens. The support of the `[string, boolean]` format in `addTokens` method is removed. ([#202](https://github.com/huggingface/tokenizers/pull/202))
- The `addSpecialTokens` option for `BertWordpieceTokenizer` has been removed, and must now be passed to `encode` and `encodeBatch` functions ([7dd2400](https://github.com/huggingface/tokenizers/commit/7dd24002148a452f4d9fc55966e181c2dc699203)) ([#193](https://github.com/huggingface/tokenizers/pull/193))

### Features

- `encode` and `encodeBatch` methods on `BaseTokenizer` now take a new optional argument, specifying whether to add the special tokens (activated by default) ([#193](https://github.com/huggingface/tokenizers/pull/193))
- Methods `decode` and `decodeBatch` exposed in `BaseTokenizer` instances ([#184](https://github.com/huggingface/tokenizers/pull/184))
- The `fromFiles` methods for `BPE` and `WordPiece` models are now `async` ([#184](https://github.com/huggingface/tokenizers/pull/184))
- Big improvements in speed for BPE (both training and tokenization) ([#165](https://github.com/huggingface/tokenizers/pull/165))
- `ByteLevel` is also a `PostProcessor` now and handles trimming the offsets if activated. This avoids the unintuitive inclusion of the whitespaces in the produced offsets, even if these whitespaces are part of the actual token. It has been added to `ByteLevelBPETokenizer` but it is off by default. ([#188](https://github.com/huggingface/tokenizers/pull/188))
- New `postProcess`, `encodeTokenized`, `encodeTokenizedBatch` and `normalize` methods on `BaseTokenizer` ([#200](https://github.com/huggingface/tokenizers/pull/200)) ([2aeae55](https://github.com/huggingface/tokenizers/commit/2aeae555e22ac58b11b4956aa3f601bb168e8c3f))
- New `mergeEncodings` static method on `Encoding` class ([#200](https://github.com/huggingface/tokenizers/pull/200)) ([0408567](https://github.com/huggingface/tokenizers/commit/0408567f23d938952f45192a3eff54d48f828882))
- New `wordIndexes` getter and new `charToToken`, `charToTokenOffsets`, `charToWordOffsets` and `tokenToWordOffsets` helper functions on `Encoding` instances ([#200](https://github.com/huggingface/tokenizers/pull/200)) ([ce3cf78](https://github.com/huggingface/tokenizers/commit/ce3cf78ea5423d483895f51f77ff0c7df07f9b0a))

### Fixes

- Fix `longest_first` truncation strategy ([#174](https://github.com/huggingface/tokenizers/issues/174))
- Fix options names in `BPE.fromFiles` ([306f427](https://github.com/huggingface/tokenizers/commit/35540d2e0715e88299f8f04f842e23b5a306f427))
- Actually expose `save` method in `Model` ([ddcf8e8](https://github.com/huggingface/tokenizers/commit/3d143a911bde8d15e1431156fe3cf7676ddcf8e8))
- The errors in async functions are now typed ([7aa6c13](https://github.com/huggingface/tokenizers/commit/4510ea5ce37d84754bb782a99353ac5627aa6c13))
- Trim the decoded string in `bpeDecoder` used by `BPETokenizer` ([#205](https://github.com/huggingface/tokenizers/issues/205)) ([3f4a6b7](https://github.com/huggingface/tokenizers/commit/3f4a6b746b921f339de3279d073b29e019ee2e5a))

# [0.5.0](https://github.com/huggingface/tokenizers/compare/node-v0.4.1...node-v0.5.0) (2020-02-27)

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
- Fix a bug in `ByteLevelBPETokenizer` that caused offsets to be wrong if a char got split up in multiple bytes ([#156](https://github.com/huggingface/tokenizers/pull/156))

## [0.4.1](https://github.com/huggingface/tokenizers/compare/node-v0.4.0...node-v0.4.1) (2020-02-11)

### Fixes

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

### Fixes

- Correctly truncate with `OnlyFirst` and `OnlySecond` strategies ([#108](https://github.com/huggingface/tokenizers/issues/108)) ([6d532fe](https://github.com/huggingface/tokenizers/commit/6d532fedb1d3626328828304a5c39807733d2fa1))
- Fix default special tokens in `BertWordPieceTokenizer` ([10e2d28](https://github.com/huggingface/tokenizers/commit/10e2d286caf517f0977c04cf8e1924aed90403c9))
- Fix return type of `getSpecialTokensMask` on `Encoding` ([9770be5](https://github.com/huggingface/tokenizers/commit/9770be566175dc9c44dd7dcaa00a57d0e4ca632b))
- Actually add special tokens in tokenizers implementations ([acef252](https://github.com/huggingface/tokenizers/commit/acef252dacc43adc414175cfc325668ad1488753))
