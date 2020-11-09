import datasets
from tokenizers import normalizers, pre_tokenizers, Tokenizer, models, trainers
import tqdm


def main():
    dataset = datasets.load_dataset("wikitext", "wikitext-103-raw-v1")
    bpe_tokenizer = Tokenizer(models.BPE())
    bpe_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    bpe_tokenizer.normalizer = normalizers.Lowercase()
    trainer = trainers.BpeTrainer(show_progress=True)

    def data_iterator():
        L = len(dataset["train"])
        B = 30000
        for i in tqdm.tqdm(range(0, L, B)):
            batch = dataset["train"][i : i + B]
            # print("H", end="", flush=True)
            yield "".join(batch["text"])

    bpe_tokenizer.train_from_iterator(trainer, data_iterator())


def main2():
    trainer = trainers.BpeTrainer(show_progress=True)
    bpe_tokenizer = Tokenizer(models.BPE())
    bpe_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    bpe_tokenizer.normalizer = normalizers.Lowercase()
    bpe_tokenizer.train(trainer, ["/home/nicolas/data/wikitext-103-raw/wiki.train.raw"])


if __name__ == "__main__":
    main()
