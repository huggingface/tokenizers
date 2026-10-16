import pytest

from tokenizers import Padding, Truncation

TEXTS = ["Hello there, how are you?", "naïve café", "Hello<|eot_id|>world", ""]


@pytest.mark.parametrize("text", TEXTS)
def test_tokenize_returns_encoding_tokens(gpt2, llama3, bert, text):
    for tokenizer in (gpt2, llama3, bert):
        assert tokenizer.tokenize(text) == tokenizer.encode(text).tokens


def test_tokenize_forwards_add_special_tokens(llama3):
    encoding = llama3.encode("Hello there", add_special_tokens=False)

    assert llama3.tokenize("Hello there", add_special_tokens=False) == encoding.tokens
    assert llama3.tokenize("Hello there") != encoding.tokens


def test_tokenize_forwards_padding(wiki):
    padding = Padding(length=4, pad_id=3)
    encoding = wiki.encode("Hello there", padding=padding)

    assert wiki.tokenize("Hello there", padding=padding) == encoding.tokens
    assert len(wiki.tokenize("Hello there")) == 2


def test_tokenize_forwards_truncation(gpt2):
    truncation = Truncation(max_length=2)
    encoding = gpt2.encode("Hello there, how are you?", truncation=truncation)

    assert gpt2.tokenize("Hello there, how are you?", truncation=truncation) == encoding.tokens
    assert len(gpt2.tokenize("Hello there, how are you?")) > 2
