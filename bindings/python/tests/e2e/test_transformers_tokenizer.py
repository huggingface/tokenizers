"""Layer 2: the transformers tokenizer API.

`AutoTokenizer` wraps a `tokenizers.Tokenizer` (its "backend") and funnels
every call below into it. This is the API most users actually type, so it is
the contract the bindings have to serve. The heavier scenarios are lifted
from the maintained example scripts in transformers' examples/pytorch/ — the
canonical data-preparation recipes users copy.
"""

import pytest
import tokenizers
from transformers import AutoTokenizer

from .conftest import TINY_GPT2, TINY_T5


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


def test_qa_features_with_overflow_and_stride(bert):
    # The question-answering recipe (run_qa.py, prepare_train_features): a
    # long context becomes several overlapping features, each carrying
    # offsets into the original string and sequence_ids to tell question
    # from context — everything needed to label answer spans for training.
    filler = "The quick brown fox jumps over the lazy dog. " * 12
    context = filler + "The Eiffel Tower is located in Paris." + filler
    answer = "Paris"
    start_char = context.index(answer)
    end_char = start_char + len(answer)

    features = bert(
        ["Where is the Eiffel Tower?"],
        [context],
        truncation="only_second",  # never truncate the question
        max_length=64,
        stride=32,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )

    n = len(features["input_ids"])
    assert n > 1
    assert features["overflow_to_sample_mapping"] == [0] * n

    recovered = []
    for i in range(n):
        seq_ids = features.sequence_ids(i)
        offsets = features["offset_mapping"][i]
        token_start = seq_ids.index(1)
        token_end = len(seq_ids) - 1
        while seq_ids[token_end] != 1:
            token_end -= 1
        if not (offsets[token_start][0] <= start_char and offsets[token_end][1] >= end_char):
            continue  # the answer is outside this feature's window
        # Walk in to the answer's exact token span, as run_qa does to label
        # training data, then read it back out of the source string.
        while token_start < len(offsets) and offsets[token_start][0] <= start_char:
            token_start += 1
        while offsets[token_end][1] >= end_char:
            token_end -= 1
        recovered.append(context[offsets[token_start - 1][0] : offsets[token_end + 1][1]])

    # The stride windows overlap, so several features see the answer and all
    # recover it exactly; the far-away windows don't contain it at all.
    assert recovered
    assert set(recovered) == {answer}
    assert len(recovered) < n


def test_ner_labels_align_through_word_ids(bert):
    # The token-classification recipe (run_ner.py, tokenize_and_align_labels):
    # one label per word in, one label per token out — the word's label on its
    # first sub-token, -100 on continuations and special tokens.
    words = ["My", "name", "is", "Sylvain", "and", "I", "work", "at", "HuggingFace"]
    word_labels = [0, 0, 0, 1, 0, 0, 0, 0, 2]

    encoding = bert([words], is_split_into_words=True)

    word_ids = encoding.word_ids(batch_index=0)
    labels = []
    previous = None
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != previous:
            labels.append(word_labels[word_idx])
        else:
            labels.append(-100)
        previous = word_idx

    assert len(labels) == len(encoding["input_ids"][0])
    assert labels[0] == labels[-1] == -100
    # Each word contributes its label exactly once, in order.
    assert [label for label in labels if label != -100] == word_labels
    # The alignment is only interesting if some word split into sub-tokens
    # ("Sylvain" and "HuggingFace" do).
    assert len(word_ids) > len(words) + 2


def test_seq2seq_targets_via_text_target():
    # The translation recipe (run_translation.py, preprocess_function):
    # text_target= tokenizes with the target-side rules, and the label ids
    # end with EOS so generation learns where to stop.
    tok = AutoTokenizer.from_pretrained(TINY_T5)

    inputs = tok(["translate English to German: Hello"], max_length=32, truncation=True)
    labels = tok(text_target=["Hallo"], max_length=32, truncation=True)
    inputs["labels"] = labels["input_ids"]

    assert inputs["labels"][0][-1] == tok.eos_token_id
    assert tok.decode(inputs["labels"][0], skip_special_tokens=True) == "Hallo"


def test_train_new_from_iterator(bert):
    # Retrain the same pipeline on a new corpus — transformers drives the
    # tokenizers trainers under the hood.
    corpus = ["the cat sat on the mat"] * 20

    new_tok = bert.train_new_from_iterator(corpus, vocab_size=60)

    assert new_tok.is_fast
    assert len(new_tok) <= 60
    ids = new_tok("the cat")["input_ids"]
    assert new_tok.batch_decode([ids], skip_special_tokens=True) == ["the cat"]
