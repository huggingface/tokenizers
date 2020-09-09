# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- [#236]: Fix a bug with offsets being shifted when there are sub-sequences (Usually with
special tokens and/or added tokens in the sequence).
- [#286]: Fix various crash when training a BPE model
- [#309]: Fixed a few bugs related to additional vocabulary/tokens
- [#363]: Fix panic from unwrapping `File::open` in `count_words`

### Changed
- [#234]: Completely changed the alignement mappings available on `Encoding`. Previous mappings
were misleading and only providing offsets. New ones provide methods to easily convert between
`char` or `word` (input space) and `token` (output space)
- [#236]: `AddedToken` with special options like `rstrip` will keep the matched whitespaces
in the textual representation of the token, exposed in `tokens` on the `Encoding`. The ID stays
the same as usual. This fixes the offsets for said tokens.
- [#236]: Offsets are now converted back to the original referential before we merge the
sub-sequences together and then do the post-processing. This also fixes some offsets bugs.
- [#236]: ByteLevel PostProcessor now uses the `add_prefix_space` attribute to determine how to
trim offsets.
- Improved `TruncationError` to handle cases where provided max length is too low.
- [#249]: `encode` and `encode_batch` input has been greatly improved, and it now also accept
pre-tokenized inputs.
- Improved `TruncationError` to handle cases where provided max length is too low.
- [#276]: Improve BPE training speeds, by reading files sequentially, but parallelizing the
processing of each file
- [#280]: Use `onig` for byte-level pre-tokenization to remove all the differences with the original
implementation from GPT-2
- [#309]: Improved the management of the additional vocabulary. This introduces an option
`normalized`, controlling whether a token should be extracted from the normalized version of the
input text.
- [#330]: BertNormalizer now keeps the same behavior than the original implementation when
`strip_accents` is not specified.
- [#355]: Tokenizer does not use any dynamic dispatch anymore.
- [#377]: Use byte offsets everywhere (instead of the char offsets)

### Added
- [#236]: RobertaProcessing is now also taking care of trimming offsets, and works just as ByteLevel
on this front.
- [#272]: Serialization of the `Tokenizer` and all the parts (`PreTokenizer`, `Normalizer`, ...)
using serde. It is now easy to save/load an entire tokenizer.
- [#289]: Ability to pad to a multiple of a specified value. This is especially useful to ensure
activation of the Tensor Cores, while ensuring padding to a multiple of 8.
- [#298]: Ability to get the currently set truncation/padding params
- [#311]: Ability to enable/disable the parallelism using the `TOKENIZERS_PARALLELISM` environment
variable.
- [#403]: Add `TemplateProcessing` `PostProcessor`.

### How to migrate
- Replace any `XXX_to_YYY_offsets()` method call by any of the new ones.
- Specify the `add_prefix_space` and `trim_offsets` options on `RobertaProcessing` if you don't
want the offsets trimmed out.
- Any custom `PostProcessor` now handles offsets relative to the original string (as opposed to the
normalized one).

## [0.10.1]

### Fixed
- [#226]: Fix the word indexes when there are special tokens

## [0.10.0]

### Changed
- [#222]: All Tokenizer's subparts must now be `Send + Sync`

### Added
- [#208]: Ability to retrieve the vocabulary from the `Tokenizer` & `Model`

### Fixed
- [#205]: Trim the decoded string in `BPEDecoder`
- [b770f36]: Fix a bug with added tokens generated IDs

## [0.9.0]

### Changed
- Only one progress bar while reading files during training. This is better for use-cases with
a high number of files as it avoids having too many progress bars on screen. Also avoids reading the
size of each file before starting to actually read these files, as this process could take really
long.
- [#190]: Improved BPE and WordPiece builders
- [#193]: `encode` and `encode_batch` now take a new argument, specifying whether we should add the
special tokens
- [#197]: The `NormalizedString` has been removed from the `Encoding`. It is now possible to
retrieve it by calling `normalize` on the `Tokenizer`. This brings a reduction of 70% of the memory
footprint
- [#197]: The `NormalizedString` API has been improved. It is now possible to retrieve parts of both
strings using both "normalized" or "original" offsets
- [#197]: The offsets provided on `Encoding` are now relative to the original string, and not the
normalized one anymore
- `AddedToken` are now used for both `add_special_tokens` and `add_tokens`. Also, these AddedToken
have more options to allow various behaviors.

### Added
- [#188]: `impl PostProcessor for ByteLevel`: Handles trimming the offsets if activated. This avoids
the unintuitive inclusion of the whitespaces in the produced offsets, even if these whitespaces are
part of the actual token
- More alignment mappings on the `Encoding`.
- `post_process` can be called on the `Tokenizer`

### Fixed
- [#193]: Fix some issues with the offsets being wrong with the `ByteLevel` BPE:
	- when `add_prefix_space` is activated
	- [#156]: when a Unicode character gets split-up in multiple byte-level characters
- Fix a bug where offsets were wrong when there was any added tokens in the sequence being encoded.
- [#175]: Fix a bug that prevented the addition of more than a certain amount of tokens (even if not
advised, but that's not the question)

### How to migrate
- Add the `ByteLevel` `PostProcessor` to your byte-level BPE tokenizers if relevant.

## [0.8.0]

### Changed
- [#165]: Big improvements in speed for BPE (Both training and tokenization)

### Fixed
- [#163]: Do not open all files directly while training
- [#156]: There was a bug in ByteLevel PreTokenizer that caused offsets to be wrong if a char got
split up in multiple bytes
- [#174]: The `LongestFirst` truncation strategy had a bug

[#403]: https://github.com/huggingface/tokenizers/pull/403
[#377]: https://github.com/huggingface/tokenizers/pull/377
[#355]: https://github.com/huggingface/tokenizers/pull/355
[#363]: https://github.com/huggingface/tokenizers/pull/363
[#330]: https://github.com/huggingface/tokenizers/pull/330
[#311]: https://github.com/huggingface/tokenizers/pull/311
[#309]: https://github.com/huggingface/tokenizers/pull/309
[#298]: https://github.com/huggingface/tokenizers/pull/298
[#289]: https://github.com/huggingface/tokenizers/pull/289
[#286]: https://github.com/huggingface/tokenizers/pull/286
[#280]: https://github.com/huggingface/tokenizers/pull/280
[#276]: https://github.com/huggingface/tokenizers/pull/276
[#272]: https://github.com/huggingface/tokenizers/pull/272
[#249]: https://github.com/huggingface/tokenizers/pull/249
[b770f36]: https://github.com/huggingface/tokenizers/commit/b770f364280af33efeffea8f0003102cda8cf1b7
[#236]: https://github.com/huggingface/tokenizers/pull/236
[#234]: https://github.com/huggingface/tokenizers/pull/234
[#226]: https://github.com/huggingface/tokenizers/pull/226
[#222]: https://github.com/huggingface/tokenizers/pull/222
[#208]: https://github.com/huggingface/tokenizers/pull/208
[#205]: https://github.com/huggingface/tokenizers/issues/205
[#197]: https://github.com/huggingface/tokenizers/pull/197
[#193]: https://github.com/huggingface/tokenizers/pull/193
[#190]: https://github.com/huggingface/tokenizers/pull/190
[#188]: https://github.com/huggingface/tokenizers/pull/188
[#175]: https://github.com/huggingface/tokenizers/issues/175
[#174]: https://github.com/huggingface/tokenizers/issues/174
[#165]: https://github.com/huggingface/tokenizers/pull/165
[#163]: https://github.com/huggingface/tokenizers/issues/163
[#156]: https://github.com/huggingface/tokenizers/pull/156
