import pytest

from tokenizers import Padding

TEXTS = ["Hello there, how are you?", "naïve café", "Hello<|eot_id|>world", ""]


@pytest.mark.parametrize("text", TEXTS)
def test_tokenize_is_encode_then_decode_tokens(gpt2, llama3, bert, text):
    for tokenizer in (gpt2, llama3, bert):
        assert tokenizer.tokenize(text) == tokenizer.decode_tokens(tokenizer.encode(text).ids)


def test_tokenize_forwards_add_special_tokens(llama3):
    encoding = llama3.encode("Hello there", add_special_tokens=False)

    assert llama3.tokenize("Hello there", add_special_tokens=False) == llama3.decode_tokens(encoding.ids)
    assert llama3.tokenize("Hello there") != llama3.decode_tokens(encoding.ids)


def test_tokenize_forwards_padding(wiki):
    padding = Padding(length=4, pad_id=3)
    encoding = wiki.encode("Hello there", padding=padding)

    assert wiki.tokenize("Hello there", padding=padding) == wiki.decode_tokens(encoding.ids)
    assert len(wiki.tokenize("Hello there")) == 2
