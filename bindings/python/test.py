from tokenizers import ByteLevelBPETokenizer
from tokenizers import pre_tokenizers, models, Tokenizer, trainers

tokenizer = Tokenizer(models.Unigram())
tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
trainer = trainers.UnigramTrainer(
        vocab_size=400000000,
                         show_progress=True,
                         special_tokens=["<s>", "<pad>", "</s>", "<unk>", "mask"]
                         )
tokenizer.train(["data/big.txt"], trainer)

