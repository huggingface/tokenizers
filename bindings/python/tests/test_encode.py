import pytest

from tokenizers import Encoding

# Every id in this file comes from released tokenizers 0.23.1 on the same fixture


@pytest.mark.parametrize(
    ("text", "ids"),
    [
        ("Hello there", [15496, 612]),
        ("Hello there, how are you today?", [15496, 612, 11, 703, 389, 345, 1909, 30]),
        ("unbelievably", [403, 6667, 11203, 1346]),
        ("line one\nline two", [1370, 530, 198, 1370, 734]),
        ("   ", [220, 220, 220]),
    ],
)
def test_gpt2_encode(gpt2, text, ids):
    encoding = gpt2.encode(text)

    assert encoding.ids == ids
    assert len(encoding) == len(ids)
    assert encoding.type_ids == [0] * len(ids)
    assert encoding.attention_mask == [1] * len(ids)


def test_leading_space(gpt2):
    assert gpt2.encode("Hello there").ids == [15496, 612]
    assert gpt2.encode(" Hello there").ids == [18435, 612]


@pytest.mark.parametrize(
    ("text", "ids"),
    [
        ("naïve café", [2616, 38776, 40304]),
        ("日本語", [33768, 98, 17312, 105, 45739, 252]),
        ("👋🏽", [41840, 233, 8582, 237, 121]),
    ],
)
def test_gpt2_multibyte_encode(gpt2, text, ids):
    assert gpt2.encode(text).ids == ids


# 128000 is `<|begin_of_text|>`, prepended by the post-processor.
@pytest.mark.parametrize(
    ("text", "ids"),
    [
        ("Hello there, how are you today?", [9906, 1070, 11, 1268, 527, 499, 3432, 30]),
        ("unbelievably", [359, 32898, 89234]),
        ("line one\nline two", [1074, 832, 198, 1074, 1403]),
        ("   ", [262]),
        ("naïve café", [3458, 38672, 588, 53050]),
        ("日本語", [102433, 102158]),
        ("👋🏽", [9468, 239, 233, 9468, 237, 121]),
    ],
)
def test_llama3_encode(llama3, text, ids):
    encoding = llama3.encode(text)

    assert encoding.ids == [128000, *ids]
    assert encoding.type_ids == [0] * (len(ids) + 1)
    assert encoding.attention_mask == [1] * (len(ids) + 1)
    assert llama3.encode(text, add_special_tokens=False).ids == ids


def test_bert_specials(bert):
    encoding = bert.encode("Hello there")

    assert encoding.ids == [1, 27462, 7495, 2]
    assert encoding.type_ids == [0, 0, 0, 0]
    assert encoding.attention_mask == [1, 1, 1, 1]
    assert bert.encode("Hello there", add_special_tokens=False).ids == [27462, 7495]


def test_special_tokens(gpt2, bert, llama3):
    assert gpt2.encode("Hello<|endoftext|>world").ids == [15496, 50256, 6894]
    assert bert.encode("Hello [SEP] there").ids == [1, 27462, 2, 7495, 2]
    assert llama3.encode("Hello<|eot_id|>world").ids == [128000, 9906, 128009, 14957]


def test_empty(gpt2, bert, llama3):
    assert gpt2.encode("").ids == []
    assert len(gpt2.encode("")) == 0
    assert bert.encode("").ids == [1, 2]
    assert bert.encode("", add_special_tokens=False).ids == []
    assert llama3.encode("").ids == [128000]
    assert llama3.encode("", add_special_tokens=False).ids == []


def test_encode_batch(gpt2):
    texts = [f"Sentence number {n}. " * (n % 17 + 1) for n in range(300)]

    batch = gpt2.encode_batch(texts)

    assert len(batch) == len(texts)
    assert all(isinstance(encoding, Encoding) for encoding in batch)
    assert [e.ids for e in batch] == [gpt2.encode(text).ids for text in texts]


def test_encode_batch_special_tokens(bert):
    batch = bert.encode_batch(["Hello", "there"], add_special_tokens=False)

    assert [e.ids for e in batch] == [[27462], [7495]]


def test_encode_batch_empty(gpt2):
    assert gpt2.encode_batch([]) == []
