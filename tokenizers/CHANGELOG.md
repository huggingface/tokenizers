# v0.8.0

## Changes:
- Big improvements in speed for BPE (Both training and tokenization) ([#165](https://github.com/huggingface/tokenizers/pull/165))

## Fixes:
- Do not open all files directly while training ([#163](https://github.com/huggingface/tokenizers/issues/163))
- There was a bug in ByteLevel PreTokenizer that caused offsets to be wrong if a char got split up
in multiple bytes. (cf [#156](https://github.com/huggingface/tokenizers/pull/156))
- The `LongestFirst` truncation strategy had a bug ([#174](https://github.com/huggingface/tokenizers/issues/174))
