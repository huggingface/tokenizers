# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0-rc5]

### Changed
- Only one progress bar while reading files during training. This is better for use-cases with
a high number of files as it avoids having too many progress bars on screen. Also avoids reading the
size of each file before starting to actually read these files, as this process could take really
long.
- [#193]: `encode` and `encode_batch` now take a new optional argument, specifying whether we
should add the special tokens. This is activated by default.
- [#197]: `original_str` and `normalized_str` have been removed from the `Encoding` returned by
`encode` and `encode_batch`. This brings a reduction of 70% of the memory footprint.
- [#197]: The offsets provided on `Encoding` are now relative to the original string, and not the
normalized one anymore.
- The added token given to `add_special_tokens` or `add_tokens` on a `Tokenizer`, or while using
`train(special_tokens=...)` can now be instances of `AddedToken` to provide more control over these
tokens.
- [#136] Updated Pyo3 version
- [#136] Static methods `Model.from_files` and `Model.empty` are removed in favor of using
constructors.

### Added
- [#188]: `ByteLevel` is also a `PostProcessor` now and handles trimming the offsets if activated.
This avoids the unintuitive inclusion of the whitespaces in the produced offsets, even if these
whitespaces are part of the actual token.
It has been added to `ByteLevelBPETokenizer` but it is off by default (`trim_offsets=False`).
- [#236]: `RobertaProcessing` also handles trimming the offsets.
- [#234]: New alignment mappings on the `Encoding`. Provide methods to easily convert between `char`
or `word` (input space) and `token` (output space).
- `post_process` can be called on the `Tokenizer`
- [#208]: Ability to retrieve the vocabulary from the `Tokenizer` with
`get_vocab(with_added_tokens: bool)`
- [#136] Models can now be instantiated through object constructors.

### Fixed
- [#193]: Fix some issues with the offsets being wrong with the `ByteLevel` BPE:
	- when `add_prefix_space=True`
	- [#156]: when a Unicode character gets split-up in multiple byte-level characters
- Fix a bug where offsets were wrong when there was any added tokens in the sequence being encoded.
- [#175]: Fix a bug that prevented the addition of more than a certain amount of tokens (even if
not advised, but that's not the question).
- [#205]: Trim the decoded string in `BPEDecoder` used by `CharBPETokenizer`

### How to migrate
- Add the `ByteLevel` `PostProcessor` to your byte-level BPE tokenizers if relevant. If you are
using `ByteLevelBPETokenizer`, this option is disabled by default (`trim_offsets=False`).
- `BertWordPieceTokenizer` option to `add_special_tokens` must now be given to `encode` or
`encode_batch`
- Access to the `original_str` on the `Encoding` has been removed. The original string is the input
of `encode` so it didn't make sense to keep it here.
- No need to call `original_str.offsets(offsets[N])` to convert offsets to the original string. They
are now relative to the original string by default.
- Access to the `normalized_str` on the `Encoding` has been removed. Can be retrieved by calling
`normalize(sequence)` on the `Tokenizer`
- Change `Model.from_files` and `Model.empty` to use constructor. The model constructor should take
the same arguments as the old methods. (ie `BPE(vocab, merges)` or `BPE()`)

## [0.6.0]

### Changed
- [#165]: Big improvements in speed for BPE (Both training and tokenization)

### Fixed
- [#160]: Some default tokens were missing from `BertWordPieceTokenizer`
- [#156]: There was a bug in ByteLevel PreTokenizer that caused offsets to be wrong if a char got
split up in multiple bytes.
- [#174]: The `longest_first` truncation strategy had a bug

## [0.5.2]
- [#163]: Do not open all files directly while training

### Fixed
- We introduced a bug related to the saving of the WordPiece model in 0.5.1: The `vocab.txt` file
was named `vocab.json`. This is now fixed.
- The `WordLevel` model was also saving its vocabulary to the wrong format.

## [0.5.1]

### Changed
- `name` argument is now optional when saving a `Model`'s vocabulary. When the name is not
specified, the files get a more generic naming, like `vocab.json` or `merges.txt`.

## [0.5.0]

### Changed
- [#145]: `BertWordPieceTokenizer` now cleans up some tokenization artifacts while decoding
- [#149]: `ByteLevelBPETokenizer` now has `dropout`.
- `do_lowercase` has been changed to `lowercase` for consistency between the different tokenizers.
(Especially `ByteLevelBPETokenizer` and `CharBPETokenizer`)
- [#139]: Expose `__len__` on `Encoding`
- Improved padding performances.

### Added
- Added a new `Strip` normalizer

### Fixed
- [#145]: Decoding was buggy on `BertWordPieceTokenizer`.
- [#152]: Some documentation and examples were still using the old `BPETokenizer`

### How to migrate
- Use `lowercase` when initializing `ByteLevelBPETokenizer` or `CharBPETokenizer` instead of
`do_lowercase`.

## [0.4.2]

### Fixed
- [#137]: Fix a bug in the class `WordPieceTrainer` that prevented `BertWordPieceTokenizer` from
being trained.

## [0.4.1]

### Fixed
- [#134]: Fix a bug related to the punctuation in BertWordPieceTokenizer

## [0.4.0]

### Changed
- [#131]: Replaced all .new() class methods by a proper __new__ implementation
- Improved typings

### How to migrate
- Remove all `.new` on all classe instanciations

## [0.3.0]

### Changed
- BPETokenizer has been renamed to CharBPETokenizer for clarity.
- Improve truncation/padding and the handling of overflowing tokens. Now when a sequence gets
truncated, we provide a list of overflowing `Encoding` that are ready to be processed by a language
model, just as the main `Encoding`.
- Provide mapping to the original string offsets using:
```
output = tokenizer.encode(...)
print(output.original_str.offsets(output.offsets[3]))
```
- [#99]: Exposed the vocabulary size on all tokenizers

### Added
- Added `CharDelimiterSplit`: a new `PreTokenizer` that allows splitting sequences on the given
delimiter (Works like `.split(delimiter)`)
- Added `WordLevel`: a new model that simply maps `tokens` to their `ids`.

### Fixed
- Fix a bug with IndexableString
- Fix a bug with truncation

### How to migrate
- Rename `BPETokenizer` to `CharBPETokenizer`
- `Encoding.overflowing` is now a List instead of a `Optional[Encoding]`

## [0.2.1]

### Fixed
- Fix a bug with the IDs associated with added tokens.
- Fix a bug that was causing crashes in Python 3.5

[#236]: https://github.com/huggingface/tokenizers/pull/236
[#234]: https://github.com/huggingface/tokenizers/pull/234
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
[#160]: https://github.com/huggingface/tokenizers/issues/160
[#156]: https://github.com/huggingface/tokenizers/pull/156
[#152]: https://github.com/huggingface/tokenizers/issues/152
[#149]: https://github.com/huggingface/tokenizers/issues/149
[#145]: https://github.com/huggingface/tokenizers/issues/145
[#139]: https://github.com/huggingface/tokenizers/issues/139
[#137]: https://github.com/huggingface/tokenizers/issues/137
[#134]: https://github.com/huggingface/tokenizers/issues/134
[#131]: https://github.com/huggingface/tokenizers/issues/131
[#99]: https://github.com/huggingface/tokenizers/pull/99
