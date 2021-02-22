# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- [#519]: Add a `WordLevelTrainer` used to train a `WordLevel` model
- [#533]: Add support for conda builds
- [#542]: Add Split pre-tokenizer to easily split using a pattern
- [#544]: Ability to train from memory. This also improves the integration with `datasets`

### Changed
- [#509]: Automatically stubbing the `.pyi` files
- [#519]: Each `Model` can return its associated `Trainer` with `get_trainer()`
- [#530]: The various attributes on each component can be get/set (ie.
`tokenizer.model.dropout = 0.1`)
- [#538]: The API Reference has been improved and is now up-to-date.

## Fixed
- [#519]: During training, the `Model` is now trained in-place. This fixes several bugs that were
forcing to reload the `Model` after a training.
- [#539]: Fix `BaseTokenizer` enable_truncation docstring

## [0.9.4]

### Fixed
- [#492]: Fix `from_file` on `BertWordPieceTokenizer`
- [#498]: Fix the link to download `sentencepiece_model_pb2.py`
- [#500]: Fix a typo in the docs quicktour

### Changed
- [#506]: Improve Encoding mappings for pairs of sequence

## [0.9.3]

### Fixed
- [#470]: Fix hanging error when training with custom component
- [#476]: TemplateProcessing serialization is now deterministic
- [#481]: Fix SentencePieceBPETokenizer.from_files

### Added
- [#477]: UnicodeScripts PreTokenizer to avoid merges between various scripts
- [#480]: Unigram now accepts an `initial_alphabet` and handles `special_tokens` correctly

## [0.9.2]

### Fixed
- [#464]: Fix a problem with RobertaProcessing being deserialized as BertProcessing

## [0.9.1]

### Fixed
- [#459]: Fix a problem with deserialization

## [0.9.0]

### Fixed
- [#362]: Fix training deadlock with Python components.
- [#363]: Fix a crash when calling `.train` with some non-existent files
- [#355]: Remove a lot of possible crashes
- [#389]: Improve truncation (crash and consistency)

### Added
- [#379]: Add the ability to call `encode`/`encode_batch` with numpy arrays
- [#292]: Support for the Unigram algorithm
- [#378], [#394], [#416], [#417]: Many new Normalizer and PreTokenizer
- [#403]: Add `TemplateProcessing` `PostProcessor`.
- [#420]: Ability to fuse the "unk" token in BPE.

### Changed
- [#360]: Lots of improvements related to words/alignment tracking
- [#426]: Improvements on error messages thanks to PyO3 0.12

## [0.8.1]

### Fixed
- [#333]: Fix deserialization of `AddedToken`, where the content was not restored properly

### Changed
- [#329]: Improved warning and behavior when we detect a fork
- [#330]: BertNormalizer now keeps the same behavior than the original implementation when
`strip_accents` is not specified.

## [0.8.0]

### Highlights of this release
- We can now encode both pre-tokenized inputs, and raw strings. This is especially usefull when
processing datasets that are already pre-tokenized like for NER (Name Entity Recognition), and helps
while applying labels to each word.
- Full tokenizer serialization. It is now easy to save a tokenizer to a single JSON file, to later
load it back with just one line of code. That's what sharing a Tokenizer means now: 1 line of code.
- With the serialization comes the compatibility with `Pickle`! The Tokenizer, all of its components,
Encodings, everything can be pickled!
- Training a tokenizer is now even faster (up to 5-10x) than before!
- Compatibility with `multiprocessing`, even when using the `fork` start method. Since this library
makes heavy use of the multithreading capacities of our computers to allows a very fast tokenization,
this led to problems (deadlocks) when used with `multiprocessing`. This version now allows to
disable the parallelism, and will warn you if this is necessary.
- And a lot of other improvements, and fixes.

### Fixed
- [#286]: Fix various crash when training a BPE model
- [#309]: Fixed a few bugs related to additional vocabulary/tokens

### Added
- [#272]: Serialization of the `Tokenizer` and all the parts (`PreTokenizer`, `Normalizer`, ...).
This adds some methods to easily save/load an entire tokenizer (`from_str`, `from_file`).
- [#273]: `Tokenizer` and its parts are now pickable
- [#289]: Ability to pad to a multiple of a specified value. This is especially useful to ensure
activation of the Tensor Cores, while ensuring padding to a multiple of 8. Use with
`enable_padding(pad_to_multiple_of=8)` for example.
- [#298]: Ability to get the currently set truncation/padding params
- [#311]: Ability to enable/disable the parallelism using the `TOKENIZERS_PARALLELISM` environment
variable. This is especially usefull when using `multiprocessing` capabilities, with the `fork`
start method, which happens to be the default on Linux systems. Without disabling the parallelism,
the process dead-locks while encoding. (Cf [#187] for more information)

### Changed
- Improved errors generated during truncation: When the provided max length is too low are
now handled properly.
- [#249] `encode` and `encode_batch` now accept pre-tokenized inputs. When the input is pre-tokenized,
the argument `is_pretokenized=True` must be specified.
- [#276]: Improve BPE training speeds, by reading files sequentially, but parallelizing the
processing of each file
- [#280]: Use `onig` for byte-level pre-tokenization to remove all the differences with the original
implementation from GPT-2
- [#309]: Improved the management of the additional vocabulary. This introduces an option
`normalized`, controlling whether a token should be extracted from the normalized version of the
input text.

## [0.7.0]

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
- [#136]: Updated Pyo3 version
- [#136]: Static methods `Model.from_files` and `Model.empty` are removed in favor of using
constructors.
- [#239]: `CharBPETokenizer` now corresponds to OpenAI GPT BPE implementation by default.

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
- If you were using the `CharBPETokenizer` and want to keep the same behavior as before, set
`bert_normalizer=False` and `split_on_whitespace_only=True`.

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


[#544]: https://github.com/huggingface/tokenizers/pull/544
[#542]: https://github.com/huggingface/tokenizers/pull/542
[#539]: https://github.com/huggingface/tokenizers/pull/539
[#538]: https://github.com/huggingface/tokenizers/pull/538
[#533]: https://github.com/huggingface/tokenizers/pull/533
[#530]: https://github.com/huggingface/tokenizers/pull/530
[#519]: https://github.com/huggingface/tokenizers/pull/519
[#509]: https://github.com/huggingface/tokenizers/pull/509
[#506]: https://github.com/huggingface/tokenizers/pull/506
[#500]: https://github.com/huggingface/tokenizers/pull/500
[#498]: https://github.com/huggingface/tokenizers/pull/498
[#492]: https://github.com/huggingface/tokenizers/pull/492
[#481]: https://github.com/huggingface/tokenizers/pull/481
[#480]: https://github.com/huggingface/tokenizers/pull/480
[#477]: https://github.com/huggingface/tokenizers/pull/477
[#476]: https://github.com/huggingface/tokenizers/pull/476
[#470]: https://github.com/huggingface/tokenizers/pull/470
[#464]: https://github.com/huggingface/tokenizers/pull/464
[#459]: https://github.com/huggingface/tokenizers/pull/459
[#420]: https://github.com/huggingface/tokenizers/pull/420
[#417]: https://github.com/huggingface/tokenizers/pull/417
[#416]: https://github.com/huggingface/tokenizers/pull/416
[#403]: https://github.com/huggingface/tokenizers/pull/403
[#394]: https://github.com/huggingface/tokenizers/pull/394
[#389]: https://github.com/huggingface/tokenizers/pull/389
[#379]: https://github.com/huggingface/tokenizers/pull/379
[#378]: https://github.com/huggingface/tokenizers/pull/378
[#363]: https://github.com/huggingface/tokenizers/pull/363
[#362]: https://github.com/huggingface/tokenizers/pull/362
[#360]: https://github.com/huggingface/tokenizers/pull/360
[#355]: https://github.com/huggingface/tokenizers/pull/355
[#333]: https://github.com/huggingface/tokenizers/pull/333
[#330]: https://github.com/huggingface/tokenizers/pull/330
[#329]: https://github.com/huggingface/tokenizers/pull/329
[#311]: https://github.com/huggingface/tokenizers/pull/311
[#309]: https://github.com/huggingface/tokenizers/pull/309
[#292]: https://github.com/huggingface/tokenizers/pull/292
[#289]: https://github.com/huggingface/tokenizers/pull/289
[#286]: https://github.com/huggingface/tokenizers/pull/286
[#280]: https://github.com/huggingface/tokenizers/pull/280
[#276]: https://github.com/huggingface/tokenizers/pull/276
[#273]: https://github.com/huggingface/tokenizers/pull/273
[#272]: https://github.com/huggingface/tokenizers/pull/272
[#249]: https://github.com/huggingface/tokenizers/pull/249
[#239]: https://github.com/huggingface/tokenizers/pull/239
[#236]: https://github.com/huggingface/tokenizers/pull/236
[#234]: https://github.com/huggingface/tokenizers/pull/234
[#208]: https://github.com/huggingface/tokenizers/pull/208
[#205]: https://github.com/huggingface/tokenizers/issues/205
[#197]: https://github.com/huggingface/tokenizers/pull/197
[#193]: https://github.com/huggingface/tokenizers/pull/193
[#190]: https://github.com/huggingface/tokenizers/pull/190
[#188]: https://github.com/huggingface/tokenizers/pull/188
[#187]: https://github.com/huggingface/tokenizers/issues/187
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
