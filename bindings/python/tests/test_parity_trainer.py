import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

EN = ["the cat sat on the mat", "the dog ate the food", "a cat and a dog"] * 6
DE = ["die katze sitzt auf der matte", "der hund frisst das futter", "eine katze und ein hund"] * 6


def fresh():
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return tok


def test_trains_with_dev_iterators():
    tok = fresh()
    trainer = trainers.ParityBpeTrainer(num_merges=30, special_tokens=["<unk>"], show_progress=False)
    trainer.train_from_iterator(tok, [iter(EN), iter(DE)], dev_iterators=[iter(EN[:6]), iter(DE[:6])])
    assert tok.token_to_id("<unk>") is not None
    for line in (EN[0], DE[0]):
        ids = tok.encode_ids(line, add_special_tokens=False)
        assert len(ids) > 0
        assert all(tok.id_to_token(int(i)) is not None for i in ids)


def test_trains_with_ratio_targets():
    tok = fresh()
    trainer = trainers.ParityBpeTrainer(num_merges=20, show_progress=False)
    trainer.train_from_iterator(tok, [iter(EN), iter(DE)], ratio=[1.0, 1.0])
    assert tok.get_vocab_size() > 0


def test_window_variant():
    tok = fresh()
    trainer = trainers.ParityBpeTrainer(num_merges=20, variant="window", window_size=5, show_progress=False)
    trainer.train_from_iterator(tok, [iter(EN), iter(DE)], dev_iterators=[iter(EN[:6]), iter(DE[:6])])
    assert tok.get_vocab_size() > 0


def test_rejects_unknown_variant():
    with pytest.raises(ValueError, match="variant"):
        trainers.ParityBpeTrainer(variant="strict")


def test_rejects_empty_train_iterators():
    trainer = trainers.ParityBpeTrainer(num_merges=10, show_progress=False)
    with pytest.raises(ValueError, match="must not be empty"):
        trainer.train_from_iterator(fresh(), [])


def test_rejects_mismatched_dev_length():
    trainer = trainers.ParityBpeTrainer(num_merges=10, show_progress=False)
    with pytest.raises(ValueError, match="must match"):
        trainer.train_from_iterator(fresh(), [iter(EN), iter(DE)], dev_iterators=[iter(EN)])


def test_iterator_error_propagates():
    def broken():
        yield "fine"
        raise RuntimeError("boom")

    trainer = trainers.ParityBpeTrainer(num_merges=10, show_progress=False)
    with pytest.raises(RuntimeError, match="boom"):
        trainer.train_from_iterator(fresh(), [broken(), iter(DE)], ratio=[1.0, 1.0])


def test_repr():
    trainer = trainers.ParityBpeTrainer(num_merges=100, variant="window")
    assert "num_merges=100" in repr(trainer)
    assert 'variant="window"' in repr(trainer)
