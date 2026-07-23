"""Layer 2: the transformers tokenizer API.

`AutoTokenizer` wraps a `tokenizers.Tokenizer` (its "backend") and funnels
every call below into it. This is the API most users actually type, so it is
the contract the bindings have to serve.
"""

import pytest
import tokenizers
from transformers import AutoTokenizer

from .conftest import TINY_GPT2


@pytest.fixture(scope="module")
def bert():
    return AutoTokenizer.from_pretrained("bert-base-uncased")


def test_the_backend_is_tokenizers(bert):
    assert bert.is_fast
    assert isinstance(bert.backend_tokenizer, tokenizers.Tokenizer)


def test_call_returns_model_inputs(bert):
    out = bert("Hello world")

    assert out["input_ids"] == [101, 7592, 2088, 102]
    assert out["token_type_ids"] == [0, 0, 0, 0]
    assert out["attention_mask"] == [1, 1, 1, 1]


def test_batch_with_dynamic_padding(bert):
    out = bert(["One two", "one two three four five"], padding=True)

    short, long = out["input_ids"]
    assert len(short) == len(long)
    assert short[-1] == bert.pad_token_id
    assert out["attention_mask"][0] == [1, 1, 1, 1, 0, 0, 0]
    assert out["attention_mask"][1] == [1] * 7


def test_truncation_to_max_length(bert):
    out = bert("one two three four five six seven eight", truncation=True, max_length=6)

    assert len(out["input_ids"]) == 6
    # The special-token template survives truncation.
    assert out["input_ids"][0] == bert.cls_token_id
    assert out["input_ids"][-1] == bert.sep_token_id


def test_return_tensors_pt(bert):
    import torch

    out = bert(["One two", "one two three four five"], padding=True, return_tensors="pt")

    assert out["input_ids"].shape == (2, 7)
    assert out["input_ids"].dtype == torch.int64
    assert out["attention_mask"].shape == (2, 7)


def test_pairs_get_type_ids(bert):
    out = bert("Where is Paris?", "Paris is in France.")

    assert set(out["token_type_ids"]) == {0, 1}


def test_offsets_and_word_ids_align_tokens_to_text(bert):
    # NER-style alignment: word_ids groups sub-word tokens back into words,
    # offsets locate them in the raw string. Special tokens map to nothing.
    text = "Tokenization highlights substrings."
    out = bert(text, return_offsets_mapping=True)

    word_ids = out.word_ids()
    assert word_ids[0] is None
    assert word_ids[-1] is None
    assert word_ids[1:3] == [0, 0]  # "token" + "##ization"

    for (start, end), word_id in zip(out["offset_mapping"], word_ids):
        if word_id is None:
            assert (start, end) == (0, 0)
        else:
            assert text[start:end] != ""


def test_batch_decode_round_trips(bert):
    texts = ["hello world", "how are you?"]
    out = bert(texts)

    assert bert.batch_decode(out["input_ids"], skip_special_tokens=True) == texts


def test_save_pretrained_round_trips(bert, tmp_path):
    bert.save_pretrained(str(tmp_path))
    reloaded = AutoTokenizer.from_pretrained(str(tmp_path))

    text = "Round-tripping through save_pretrained."
    assert reloaded(text)["input_ids"] == bert(text)["input_ids"]


def test_chat_template_renders_then_tokenizes():
    tok = AutoTokenizer.from_pretrained(TINY_GPT2)
    tok.chat_template = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}\n{% endfor %}"
    messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]

    text = tok.apply_chat_template(messages, tokenize=False)
    assert text == "<|user|>Hi\n<|assistant|>Hello!\n"
    # Byte-level BPE, so the rendered conversation round-trips exactly.
    assert tok.decode(tok(text)["input_ids"]) == text


def test_train_new_from_iterator(bert):
    # Retrain the same pipeline on a new corpus — transformers drives the
    # tokenizers trainers under the hood.
    corpus = ["the cat sat on the mat"] * 20

    new_tok = bert.train_new_from_iterator(corpus, vocab_size=60)

    assert new_tok.is_fast
    assert len(new_tok) <= 60
    ids = new_tok("the cat")["input_ids"]
    assert new_tok.batch_decode([ids], skip_special_tokens=True) == ["the cat"]
