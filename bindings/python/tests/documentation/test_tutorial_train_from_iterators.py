# flake8: noqa
import gzip
import os

import datasets
import pytest

from ..utils import data_dir, train_files


class TestTrainFromIterators:
    @staticmethod
    def get_tokenizer_trainer():
        # START init_tokenizer_trainer
        from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

        tokenizer = Tokenizer(models.Unigram())
        tokenizer.normalizer = normalizers.NFKC()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.UnigramTrainer(
            vocab_size=20000,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<PAD>", "<BOS>", "<EOS>"],
        )
        # END init_tokenizer_trainer
        trainer.show_progress = False

        return tokenizer, trainer

    @staticmethod
    def load_dummy_dataset():
        # START load_dataset
        import datasets

        dataset = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="train+test+validation")
        # END load_dataset

    @pytest.fixture(scope="class")
    def setup_gzip_files(self, train_files):
        with open(train_files["small"], "rt") as small:
            for n in range(3):
                path = f"data/my-file.{n}.gz"
                with gzip.open(path, "wt") as f:
                    f.write(small.read())

    def test_train_basic(self):
        tokenizer, trainer = self.get_tokenizer_trainer()

        # START train_basic
        # First few lines of the "Zen of Python" https://www.python.org/dev/peps/pep-0020/
        data = [
            "Beautiful is better than ugly."
            "Explicit is better than implicit."
            "Simple is better than complex."
            "Complex is better than complicated."
            "Flat is better than nested."
            "Sparse is better than dense."
            "Readability counts."
        ]
        tokenizer.train_from_iterator(data, trainer=trainer)
        # END train_basic

    def test_datasets(self):
        tokenizer, trainer = self.get_tokenizer_trainer()

        # In order to keep tests fast, we only use the first 100 examples
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        dataset = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="train[0:100]")

        # START def_batch_iterator
        def batch_iterator(batch_size=1000):
            # Only keep the text column to avoid decoding the rest of the columns unnecessarily
            tok_dataset = dataset.select_columns("text")
            for batch in tok_dataset.iter(batch_size):
                yield batch["text"]

        # END def_batch_iterator

        # START train_datasets
        tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(dataset))
        # END train_datasets

    def test_gzip(self, setup_gzip_files):
        tokenizer, trainer = self.get_tokenizer_trainer()

        # START single_gzip
        import gzip

        with gzip.open("data/my-file.0.gz", "rt") as f:
            tokenizer.train_from_iterator(f, trainer=trainer)
        # END single_gzip
        # START multi_gzip
        files = ["data/my-file.0.gz", "data/my-file.1.gz", "data/my-file.2.gz"]

        def gzip_iterator():
            for path in files:
                with gzip.open(path, "rt") as f:
                    for line in f:
                        yield line

        tokenizer.train_from_iterator(gzip_iterator(), trainer=trainer)
        # END multi_gzip
