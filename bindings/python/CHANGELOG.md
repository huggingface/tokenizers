# v0.6.0

## Changes:
- Big improvements in speed for BPE (Both training and tokenization) ([#165](https://github.com/huggingface/tokenizers/pull/165))

## Fixes:
- Some default tokens were missing from `BertWordPieceTokenizer` (cf [#160](https://github.com/huggingface/tokenizers/issues/160))
- There was a bug in ByteLevel PreTokenizer that caused offsets to be wrong if a char got split up
in multiple bytes. (cf [#156](https://github.com/huggingface/tokenizers/pull/156))
- The `longest_first` truncation strategy had a bug ([#174](https://github.com/huggingface/tokenizers/issues/174))

# v0.5.2
- Do not open all files directly while training ([#163](https://github.com/huggingface/tokenizers/issues/163))

## Fixes:
- We introduced a bug related to the saving of the WordPiece model in 0.5.1: The `vocab.txt` file was named
`vocab.json`. This is now fixed.
- The `WordLevel` model was also saving its vocabulary to the wrong format.

# v0.5.1

## Changes:
- `name` argument is now optional when saving a `Model`'s vocabulary. When the name is not specified,
the files get a more generic naming, like `vocab.json` or `merges.txt`.

# v0.5.0

## Changes:
- `BertWordPieceTokenizer` now cleans up some tokenization artifacts while decoding (cf [#145](https://github.com/huggingface/tokenizers/issues/145))
- `ByteLevelBPETokenizer` now has `dropout` (thanks @colinclement with [#149](https://github.com/huggingface/tokenizers/issues/149))
- Added a new `Strip` normalizer
- `do_lowercase` has been changed to `lowercase` for consistency between the different tokenizers. (Especially `ByteLevelBPETokenizer` and `CharBPETokenizer`)
- Expose `__len__` on `Encoding` (cf [#139](https://github.com/huggingface/tokenizers/issues/139))
- Improved padding performances.

## Fixes:
- [#145](https://github.com/huggingface/tokenizers/issues/145): Decoding was buggy on `BertWordPieceTokenizer`.
- [#152](https://github.com/huggingface/tokenizers/issues/152): Some documentation and examples were still using the old `BPETokenizer`

## How to migrate:
- Use `lowercase` when initializing `ByteLevelBPETokenizer` or `CharBPETokenizer` instead of `do_lowercase`.

# v0.4.2

## Fixes:
- Fix a bug in the class `WordPieceTrainer` that prevented `BertWordPieceTokenizer` from being trained. (cf [#137](https://github.com/huggingface/tokenizers/issues/137))

# v0.4.1

## Fixes:
- Fix a bug related to the punctuation in BertWordPieceTokenizer (Thanks to @Mansterteddy with [#134](https://github.com/huggingface/tokenizers/issues/134))

# v0.4.0

## Changes:
- Replaced all .new() class methods by a proper __new__ implementation. (Huge thanks to @ljos with [#131](https://github.com/huggingface/tokenizers/issues/131))
- Improved typings

## How to migrate:
- Remove all `.new` on all classe instanciations

# v0.3.0

## Changes:
- BPETokenizer has been renamed to CharBPETokenizer for clarity.
- Added `CharDelimiterSplit`: a new `PreTokenizer` that allows splitting sequences on the given delimiter (Works like `.split(delimiter)`)
- Added `WordLevel`: a new model that simply maps `tokens` to their `ids`.
- Improve truncation/padding and the handling of overflowing tokens. Now when a sequence gets truncated, we provide a list of overflowing `Encoding` that are ready to be processed by a language model, just as the main `Encoding`.
- Provide mapping to the original string offsets using:
```
output = tokenizer.encode(...)
print(output.original_str.offsets(output.offsets[3]))
```
- Exposed the vocabulary size on all tokenizers: [#99](https://github.com/huggingface/tokenizers/pull/99) by @kdexd

## Fixes:
- Fix a bug with IndexableString
- Fix a bug with truncation

## How to migrate:
- Rename `BPETokenizer` to `CharBPETokenizer`
- `Encoding.overflowing` is now a List instead of a `Optional[Encoding]`

# v0.2.1

## Fixes:
- Fix a bug with the IDs associated with added tokens.
- Fix a bug that was causing crashes in Python 3.5
