import numpy as np
import pytest
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

from .conftest import train_word_tokenizer


def test_all_models_construct_and_assign():
    for model in [
        models.BPE(),
        models.BPE(unk_token="<unk>", dropout=0.1, byte_fallback=True),
        models.WordPiece(unk_token="[UNK]", continuing_subword_prefix="##"),
        models.WordLevel(unk_token="[UNK]"),
        models.Unigram(),
    ]:
        tok = Tokenizer(model)
        assert type(model).__name__ in repr(tok.model)


def test_all_normalizers_construct():
    for normalizer in [
        normalizers.NFC(),
        normalizers.NFD(),
        normalizers.NFKC(),
        normalizers.NFKD(),
        normalizers.Lowercase(),
        normalizers.StripAccents(),
        normalizers.Strip(),
        normalizers.Replace("a", "b"),
        normalizers.Prepend("_"),
        normalizers.BertNormalizer(strip_accents=True, lowercase=False),
        normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase()]),
    ]:
        tok = Tokenizer(models.BPE())
        tok.normalizer = normalizer
        assert isinstance(tok.normalizer, normalizers.Normalizer)


def test_all_pre_tokenizers_construct():
    for pre_tokenizer in [
        pre_tokenizers.Whitespace(),
        pre_tokenizers.WhitespaceSplit(),
        pre_tokenizers.BertPreTokenizer(),
        pre_tokenizers.UnicodeScripts(),
        pre_tokenizers.ByteLevel(),
        pre_tokenizers.ByteLevel(use_regex=False),
        pre_tokenizers.CharDelimiterSplit(","),
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.FixedLength(length=4),
        pre_tokenizers.Punctuation(),
        pre_tokenizers.Split(" ", behavior="removed"),
        pre_tokenizers.Sequence([pre_tokenizers.Whitespace(), pre_tokenizers.Digits()]),
    ]:
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizer
        assert isinstance(tok.pre_tokenizer, pre_tokenizers.PreTokenizer)


def test_byte_level_alphabet():
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    assert len(alphabet) == 256
    assert len(set(alphabet)) == 256
    assert all(len(c) == 1 for c in alphabet)


def test_lowercase_normalizer_changes_ids():
    tok = train_word_tokenizer()
    assert tok.token_to_id("THE") is None
    before = tok.encode_ids("THE", add_special_tokens=False)
    assert [tok.id_to_token(int(i)) for i in before] == ["[UNK]"]

    tok.normalizer = normalizers.Lowercase()
    after = tok.encode_ids("THE", add_special_tokens=False)
    assert [tok.id_to_token(int(i)) for i in after] == ["the"]


def test_char_delimiter_split_effect():
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.CharDelimiterSplit(",")
    tok.train_from_iterator(["a,b", "b,c"], trainer=trainers.WordLevelTrainer(special_tokens=["[UNK]"]))
    ids = tok.encode_ids("a,c", add_special_tokens=False)
    assert [tok.id_to_token(int(i)) for i in ids] == ["a", "c"]


def test_component_assignment_invalidates_pipeline():
    tok = train_word_tokenizer()
    the_id = tok.encode_ids("the", add_special_tokens=False)
    tok.pre_tokenizer = pre_tokenizers.FixedLength(length=1)
    per_char = tok.encode_ids("the", add_special_tokens=False)
    assert len(per_char) == 3
    assert not np.array_equal(the_id, per_char)


def test_split_rejects_bad_behavior():
    with pytest.raises(ValueError):
        pre_tokenizers.Split(" ", behavior="not-a-behavior")
