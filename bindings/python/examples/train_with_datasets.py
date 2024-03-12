import datasets

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers


# Build a tokenizer
bpe_tokenizer = Tokenizer(models.BPE())
bpe_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
bpe_tokenizer.normalizer = normalizers.Lowercase()

# Initialize a dataset
dataset = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="train")


# Build an iterator over this dataset
def batch_iterator():
    batch_size = 1000
    for batch in dataset.iter(batch_size=batch_size):
        yield batch["text"]


# And finally train
bpe_tokenizer.train_from_iterator(batch_iterator(), length=len(dataset))
