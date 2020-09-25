from tokenizers import (
    Tokenizer,
    normalizers,
    pre_tokenizers,
    models,
    decoders,
    processors,
    trainers,
    AddedToken,
)


vocab_size = 100

tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = normalizers.Sequence(
    [
        normalizers.Strip(),
        normalizers.NFC(),
    ]
)
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
tokenizer.post_processor = processors.ByteLevel()
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    min_frequency=0,
    special_tokens=[
        AddedToken("<s>"),
        AddedToken("<pad>"),
        AddedToken("</s>"),
        AddedToken("<unk>"),
        AddedToken("<mask>"),
    ],
    show_progress=False,
)

tokenizer.train(trainer, ["data/small.txt"])
tokenizer.save("data/tokenizer.json")
