# v0.8.0 (not released yet)

## Fixes:
- Do not open all files directly while training ([#163](https://github.com/huggingface/tokenizers/issues/163))
- There was a bug in ByteLevel PreTokenizer that caused offsets to be wrong if a char got split up
in multiple bytes. (cf [#156](https://github.com/huggingface/tokenizers/pull/156))
