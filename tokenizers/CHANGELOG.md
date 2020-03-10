# Not released yet

## Changes:
- Keep only one progress bar while reading files during training. This is better for use-cases with
a high number of files as it avoids having too many progress bars on screen.
- Improve BPE and WordPiece builders.
- `ByteLevel` is also a `PostProcessor` now and handles trimming the offsets if activated. This
avoids the unintuitive inclusion of the whitespaces in the produced offsets, even if these
whitespaces are part of the actual token.

## Fixes:
- Fix some issues with the offsets being wrong with the `ByteLevel` BPE:
	- when `add_prefix_space` is activated
	- when a Unicode character gets split-up in multiple byte-level characters ([#156](https://github.com/huggingface/tokenizers/issues/156))

## How to migrate:
- Add the `ByteLevel` `PostProcessor` to your byte-level BPE tokenizers if relevant.

# v0.8.0

## Changes:
- Big improvements in speed for BPE (Both training and tokenization) ([#165](https://github.com/huggingface/tokenizers/pull/165))

## Fixes:
- Do not open all files directly while training ([#163](https://github.com/huggingface/tokenizers/issues/163))
- There was a bug in ByteLevel PreTokenizer that caused offsets to be wrong if a char got split up
in multiple bytes. (cf [#156](https://github.com/huggingface/tokenizers/pull/156))
- The `LongestFirst` truncation strategy had a bug ([#174](https://github.com/huggingface/tokenizers/issues/174))
