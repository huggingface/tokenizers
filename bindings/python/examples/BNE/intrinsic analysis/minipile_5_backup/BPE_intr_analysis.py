from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel, Whitespace
import datasets
import os

# Define name of test
name = "bpe_byteLevel_minipile_5_tokens_"

# Vocab length
tokens = [2**18, 2**17, 2**16, 2**15, 2**14, 2**13, 2**12, 2**11, 2**10]

# Build tokenizer
models = []
trainers = []
tokenizers = []
for vocab_len in tokens:
    models.append(BPE(unk_token="[UNK]"))
    tokenizers.append(Tokenizer(models[-1]))
    trainers.append(BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=vocab_len))
    tokenizers[-1].pre_tokenizer = ByteLevel()

# Load dataset
dataset = datasets.load_dataset("JeanKaddour/minipile", split="train").shard(
    num_shards=20, index=7
)  # .train_test_split(test_size=0.75, seed=42)["train"]


# Build an iterator over this dataset
def batch_iterator():
    batch_size = 1000
    for batch in dataset.iter(batch_size=batch_size):
        yield batch["text"]


os.mkdir("data/BPE/")
for index in range(len(tokens)):
    tokenizers[index].train_from_iterator(batch_iterator(), trainers[index], length=len(dataset))
    tokenizers[index].save(f"data/BPE/{name}{tokens[index]}.json")

    models[index].save("data/BPE/", f"{name}{tokens[index]}")
