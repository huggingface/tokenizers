from tokenizers import Tokenizer
from tokenizers.models import BNE
from tokenizers.trainers import BneTrainer
from tokenizers.pre_tokenizers import ByteLevel, Whitespace
import datasets
import os

# Define name of test
name = "bne_rest_byteLevel_minipile_5_tokens_"

# Vocab length
#tokens = [2**18, 2**17, 2**16, 2**15, 2**14, 2**13, 2**12, 2**11, 2**10]
tokens = [2**12, 2**11, 2**10]

# Build tokenizer
models = []
trainers = []
tokenizers = []
for vocab_len in tokens:
    models.append(BNE(unk_token="[UNK]"))
    tokenizers.append(Tokenizer(models[-1]))
    trainers.append(BneTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=vocab_len, max_ngram_length=48))
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


#os.mkdir("data/BNE_rest/")
for index in range(len(tokens)):
    tokenizers[index].train_from_iterator(batch_iterator(), trainers[index], length=len(dataset))
    tokenizers[index].save(f"data/BNE_rest/{name}{tokens[index]}.json")

    models[index].save("data/BNE_rest/", f"{name}{tokens[index]}")
