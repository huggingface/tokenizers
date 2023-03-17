from tokenizers import Tokenizer, models, trainers, pre_tokenizers

bpe_model = models.BPE()
tokenizer = Tokenizer(model=bpe_model)
# tokenizer.add_special_tokens(["[UNK]"])
# tokenizer.add_tokens(["~ing", "~ed"])
# tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

tokenizer.train_from_iterator(
    iterator=["test~ing lick~ing kick~ing"],
    trainer=trainers.BpeTrainer(
        special_tokens=[
            "[UNK]",
            "~ing",
            # "~ed",
        ],
    ),
)
# tokenizer.add_tokens(["~ing"])
tokenizer.save("tok.json")

assert tokenizer.token_to_id("~ing") != tokenizer.token_to_id("[UNK]")
