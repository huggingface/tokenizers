import datasets
from tokenizers import normalizers, pre_tokenizers, Tokenizer, models, trainers

# Build a tokenizer
bpe_tokenizer = Tokenizer(models.BPE())
bpe_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
bpe_tokenizer.normalizer = normalizers.Lowercase()

# Initialize a dataset
dataset = datasets.load_dataset("wikitext", "wikitext-103-raw-v1")

# Build an iterator over this dataset
def batch_iterator():
    batch_length = 1000
    for i in range(0, len(dataset["train"]), batch_length):
        yield dataset["train"][i : i + batch_length]["text"]


# And finally train
bpe_tokenizer.train_from_iterator(batch_iterator(), length=len(dataset["train"]))
