# Not released yet

## Changes:
- Keep only one progress bar while reading files during training. This is better for use-cases with
a high number of files as it avoids having too many progress bar on screen.
- Improve BPE and WordPiece builders.
- `ByteLevel` is also a `Normalizer` and handles the `add_prefix_space` option at this level now.
This fixes some issues with the offsets being wrong if this option was on.
- `ByteLevel` is also a `PostProcessor` now and handles fixing the offsets when a unicode
character get split up in a byte-level character.

## How to migrate:
- Use the `ByteLevel` as a `Normalizer` if `add_prefix_space` is required.

# v0.8.0

## Changes:
- Big improvements in speed for BPE (Both training and tokenization) ([#165](https://github.com/huggingface/tokenizers/pull/165))

## Fixes:
- Do not open all files directly while training ([#163](https://github.com/huggingface/tokenizers/issues/163))
- There was a bug in ByteLevel PreTokenizer that caused offsets to be wrong if a char got split up
in multiple bytes. (cf [#156](https://github.com/huggingface/tokenizers/pull/156))
- The `LongestFirst` truncation strategy had a bug ([#174](https://github.com/huggingface/tokenizers/issues/174))
