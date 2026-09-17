import pytest

from tokenizers import Padding

# Every token in this file is `Encoding.tokens` from released tokenizers 0.23.1 on the same fixture.


def test_gpt2_tokenize(gpt2):
    assert gpt2.tokenize("Hello there, how are you?") == ["Hello", "Ġthere", ",", "Ġhow", "Ġare", "Ġyou", "?"]
    assert gpt2.tokenize("   ") == ["Ġ", "Ġ", "Ġ"]


# A byte-level vocab entry is not always whole characters: these are the printable
# spellings of the bytes, as the tokenizer.json writes them.
def test_gpt2_tokenize_multibyte(gpt2):
    assert gpt2.tokenize("naïve café") == ["na", "Ã¯ve", "ĠcafÃ©"]
    assert gpt2.tokenize("日本語") == ["æĹ", "¥", "æľ", "¬", "èª", "ŀ"]
    assert gpt2.tokenize("👋🏽") == ["ðŁĳ", "ĭ", "ðŁ", "ı", "½"]


def test_llama3_tokenize(llama3):
    assert llama3.tokenize("Hello there") == ["<|begin_of_text|>", "Hello", "Ġthere"]
    assert llama3.tokenize("Hello there", add_special_tokens=False) == ["Hello", "Ġthere"]
    assert llama3.tokenize("日本語") == ["<|begin_of_text|>", "æĹ¥æľ¬", "èªŀ"]
    assert llama3.tokenize("Hello<|eot_id|>world") == ["<|begin_of_text|>", "Hello", "<|eot_id|>", "world"]


def test_bert_tokenize(bert):
    assert bert.tokenize("Hello there, how are you?") == [
        "[CLS]",
        "hello",
        "there",
        ",",
        "how",
        "are",
        "you",
        "?",
        "[SEP]",
    ]
    assert bert.tokenize("日本語") == ["[CLS]", "日", "##本", "##語", "[SEP]"]
    assert bert.tokenize("👋🏽") == ["[CLS]", "[UNK]", "[SEP]"]


def test_wiki_tokenize(wiki):
    assert wiki.tokenize("naïve café") == ["na", "ï", "ve", "c", "afé"]
    assert wiki.tokenize("Hello [SEP] there") == ["Hello", "[SEP]", "there"]


def test_tokenize_empty(gpt2, llama3, bert):
    assert gpt2.tokenize("") == []
    assert llama3.tokenize("") == ["<|begin_of_text|>"]
    assert llama3.tokenize("", add_special_tokens=False) == []
    assert bert.tokenize("") == ["[CLS]", "[SEP]"]


def test_tokenize_padding(wiki):
    assert wiki.tokenize("Hello there", padding=Padding(length=4, pad_id=3)) == ["Hello", "there", "[PAD]", "[PAD]"]


def test_tokenize_has_no_skip_special_tokens(gpt2):
    with pytest.raises(TypeError):
        gpt2.tokenize("Hello", skip_special_tokens=True)


@pytest.mark.parametrize("text", ["Hello there, how are you?", "naïve café", "日本語", "Hello<|eot_id|>world", ""])
def test_tokenize_is_encode_then_decode_tokens(gpt2, llama3, bert, text):
    for tokenizer in (gpt2, llama3, bert):
        assert tokenizer.tokenize(text) == tokenizer.decode_tokens(tokenizer.encode(text).ids)
