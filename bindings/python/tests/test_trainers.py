import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from .conftest import SENTENCES


def fresh(model):
    tok = Tokenizer(model)
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return tok


@pytest.mark.parametrize(
    ("model", "trainer"),
    [
        (models.BPE(), trainers.BpeTrainer(vocab_size=120, special_tokens=["<unk>"], show_progress=False)),
        (
            models.WordPiece(unk_token="[UNK]"),
            trainers.WordPieceTrainer(vocab_size=120, special_tokens=["[UNK]"], show_progress=False),
        ),
        (
            models.WordLevel(unk_token="[UNK]"),
            trainers.WordLevelTrainer(special_tokens=["[UNK]"], show_progress=False),
        ),
        (
            models.Unigram(),
            trainers.UnigramTrainer(vocab_size=64, special_tokens=["<unk>"], unk_token="<unk>", show_progress=False),
        ),
    ],
    ids=["bpe", "wordpiece", "wordlevel", "unigram"],
)
def test_each_trainer_trains(model, trainer):
    tok = fresh(model)
    tok.train_from_iterator(SENTENCES, trainer=trainer)
    assert tok.get_vocab_size() > 0
    ids = tok.encode_ids(SENTENCES[0], add_special_tokens=False)
    assert len(ids) > 0


def test_special_tokens_get_first_ids():
    tok = fresh(models.BPE())
    tok.train_from_iterator(
        SENTENCES,
        trainer=trainers.BpeTrainer(vocab_size=120, special_tokens=["<pad>", "<s>", "</s>"], show_progress=False),
    )
    assert tok.token_to_id("<pad>") == 0
    assert tok.token_to_id("<s>") == 1
    assert tok.token_to_id("</s>") == 2


def test_initial_alphabet_is_forced_in():
    tok = fresh(models.BPE())
    tok.train_from_iterator(
        SENTENCES,
        trainer=trainers.BpeTrainer(vocab_size=120, initial_alphabet=["£"], show_progress=False),
    )
    assert tok.token_to_id("£") is not None


def test_default_trainer_used_when_none_given():
    tok = fresh(models.WordLevel(unk_token="[UNK]"))
    tok.train_from_iterator(SENTENCES)
    assert tok.get_vocab_size() > 0


def test_trainer_repr():
    assert "BpeTrainer" in repr(trainers.BpeTrainer())
