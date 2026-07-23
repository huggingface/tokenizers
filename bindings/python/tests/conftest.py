from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

DATA = Path(__file__).resolve().parents[3] / "tokenizers" / "data"

SENTENCES = [
    "the quick brown fox jumps over the lazy dog",
    "pack my box with five dozen liquor jugs",
    "how vexingly quick daft zebras jump",
    "the five boxing wizards jump quickly",
] * 8


def data_file(name: str) -> Path:
    path = DATA / name
    if not path.is_file():
        pytest.skip(f"{path} missing — run `make -C ../../tokenizers fixtures bench-models data/big.txt`")
    return path


@pytest.fixture(scope="session")
def corpus():
    text = data_file("big.txt").read_text(encoding="utf-8")[:200_000]
    return [line for line in text.splitlines() if line.strip()]


@pytest.fixture(scope="session")
def gpt2_file():
    return data_file("gpt2.json")


@pytest.fixture(scope="session")
def bert_file():
    return data_file("bert-base-uncased.json")


@pytest.fixture(scope="session")
def t5_file():
    return data_file("t5-base.json")


def train_word_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(SENTENCES, trainer=trainers.WordLevelTrainer(special_tokens=["[UNK]"]))
    return tok


@pytest.fixture(scope="session")
def word_tokenizer():
    """A small trained tokenizer shared by read-only tests. Tests that mutate
    build their own with `train_word_tokenizer()`."""
    return train_word_tokenizer()
