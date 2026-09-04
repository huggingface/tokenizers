import pytest


@pytest.mark.parametrize(
    "text",
    [
        "Hello there, how are you today?",
        " Hello there",
        "naïve café",
        "日本語",
        "👋🏽",
        "line one\nline two",
        "   ",
        "",
    ],
)
def test_gpt2_decode(gpt2, text):
    assert gpt2.decode(gpt2.encode(text).ids) == text


def test_bert_decode(bert):
    assert bert.decode(bert.encode("Hello there").ids) == "hello there"
    assert bert.decode(bert.encode("Café").ids) == "cafe"
