"""Train from a Hugging Face dataset without writing it to disk —
`train_from_iterator` streams text straight into the Rust trainer, which runs
multi-threaded while the iterator is drained in 256-line gulps.

Needs the `datasets` package; downloads wikitext-2 (~12 MB) on first run."""

import datasets

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = normalizers.Lowercase()
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

dataset = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")


def texts(batch_size=1000):
    for batch in dataset.iter(batch_size=batch_size):
        yield from batch["text"]


tokenizer.train_from_iterator(texts())

print(f"trained: {tokenizer.get_vocab_size()} tokens")
ids = tokenizer.encode("the quick brown fox", add_special_tokens=False)
print([tokenizer.id_to_token(i) for i in ids])
