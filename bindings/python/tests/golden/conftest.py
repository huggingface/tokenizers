"""Golden end-to-end tests: the behavior the 1.0 bindings rewrite works toward.

Each file drives the bindings the way a real application would, one layer
further from the library than the last:

- test_direct_api.py — the `tokenizers` API itself
- test_transformers_tokenizer.py — the transformers tokenizer API, which wraps `tokenizers`
- test_transformers_inference.py — transformers inference (`generate`, `pipeline`)
- test_transformers_training.py — transformers training (`Trainer`, data collators)

Against the tokenizers wheel that transformers v5 resolves, every test passes:
`make golden-release`. Against the in-tree build (`make golden`) the failures
enumerate what the rewrite is still missing — expected until it is complete.
A test that fails on both sides is a bug in the test.

The suite runs in its own venv (transformers + torch, see requirements.txt)
and downloads a few small models from the Hub on first run. Without
transformers installed — e.g. during a plain `make test` — it skips itself.
"""

import pytest

pytest.importorskip("transformers", reason="golden tests run in a dedicated venv — use `make golden`")

# Randomly-initialized miniatures of the real architectures: full tokenizer
# and model plumbing at a few MB per download. Their outputs are gibberish,
# so tests assert mechanics (ids, shapes, round-trips), never quality.
TINY_GPT2 = "hf-internal-testing/tiny-random-gpt2"
TINY_BERT_CLS = "hf-internal-testing/tiny-random-BertForSequenceClassification"
TINY_BERT_MLM = "hf-internal-testing/tiny-random-BertForMaskedLM"
TINY_BERT_NER = "hf-internal-testing/tiny-random-BertForTokenClassification"
TINY_T5 = "hf-internal-testing/tiny-random-t5"
