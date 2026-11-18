import pytest

TEXT = "Hello there, how are you today?"
BEGIN_OF_TEXT = "<|begin_of_text|>"


@pytest.mark.parametrize("name", ["gpt2", "bert", "wiki", "llama3"])
def test_lookups_agree_with_encode(name, request):
    tokenizer = request.getfixturevalue(name)
    ids = tokenizer.encode(TEXT).ids
    tokens = tokenizer.tokenize(TEXT)

    assert [tokenizer.token_to_id(token) for token in tokens] == ids
    assert [tokenizer.id_to_token(id) for id in ids] == tokens


def test_lookup_miss(gpt2):
    assert gpt2.token_to_id("not a token") is None
    assert gpt2.id_to_token(10**6) is None


def test_vocab_includes_added_tokens_by_default(llama3):
    with_added = llama3.get_vocab()
    without_added = llama3.get_vocab(with_added_tokens=False)

    assert with_added[BEGIN_OF_TEXT] == llama3.token_to_id(BEGIN_OF_TEXT)
    assert BEGIN_OF_TEXT not in without_added
    assert llama3.get_vocab_size() == len(with_added)
    assert llama3.get_vocab_size(False) == len(without_added)
