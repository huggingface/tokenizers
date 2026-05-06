from tokenizers import Tokenizer
from tokenizers.models import BNE
from tokenizers.trainers import BneTrainer
from tokenizers.pre_tokenizers import ByteLevel, Whitespace
import datasets
import os

# Build tokenizer
model = BNE(unk_token="[UNK]")
tokenizer = Tokenizer(model)
trainer = BneTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=8000, max_ngram_length=2)
tokenizer.pre_tokenizer = ByteLevel()

# Load dataset
dataset = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="train")


# Build an iterator over this dataset
def batch_iterator():
    batch_size = 1000
    for batch in dataset.iter(batch_size=batch_size):
        yield batch["text"]

tokenizer.train_from_iterator(batch_iterator(), trainer, length=len(dataset))
tokenizer.save("data/BNE_as_BPE.json")
