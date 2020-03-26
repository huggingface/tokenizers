# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[#197]: https://github.com/huggingface/tokenizers/pull/197
[#193]: https://github.com/huggingface/tokenizers/pull/193
[#190]: https://github.com/huggingface/tokenizers/pull/190
[#188]: https://github.com/huggingface/tokenizers/pull/188
[#175]: https://github.com/huggingface/tokenizers/issues/175
[#174]: https://github.com/huggingface/tokenizers/issues/174
[#165]: https://github.com/huggingface/tokenizers/pull/165
[#163]: https://github.com/huggingface/tokenizers/issues/163
[#156]: https://github.com/huggingface/tokenizers/pull/156
