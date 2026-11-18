import pytest

from tokenizers import AddedToken, Encoding, Tokenizer, models, normalizers

# The GPT-2 and BERT fixture IDs come from released tokenizers 0.23.1.


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


def test_bert_specials(bert):
    encoding = bert.encode("Hello there")

    assert encoding.ids == [1, 27462, 7495, 2]
    assert encoding.type_ids == [0, 0, 0, 0]
    assert encoding.attention_mask == [1, 1, 1, 1]
    assert bert.encode("Hello there", add_special_tokens=False).ids == [27462, 7495]


def test_special_tokens(gpt2, bert):
    assert gpt2.encode("Hello<|endoftext|>world").ids == [15496, 50256, 6894]
    assert bert.encode("Hello [SEP] there").ids == [1, 27462, 2, 7495, 2]


def test_empty(gpt2, bert):
    assert gpt2.encode("").ids == []
    assert len(gpt2.encode("")) == 0
    assert bert.encode("").ids == [1, 2]
    assert bert.encode("", add_special_tokens=False).ids == []


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


@pytest.mark.parametrize("pair", [("Hello", "there"), ["Hello", "there"]])
def test_encode_pair(bert, pair):
    encoding = bert.encode(pair)

    assert encoding.ids == [1, 27462, 2, 7495, 2]
    assert encoding.type_ids == [0, 0, 0, 1, 1]
    assert bert.encode_batch([pair, "Hello"]) == [encoding, bert.encode("Hello")]


@pytest.mark.parametrize("pair", [["Hello"], ["Hello", "there", "again"]])
def test_encode_pair_needs_two_texts(bert, pair):
    with pytest.raises(TypeError):
        bert.encode(pair)


@pytest.mark.parametrize(("special", "normalized"), [(True, False), (False, True)])
def test_encode_untrusted_spans(special, normalized):
    spec = Tokenizer(models.BPE(vocab={"a": 0, "b": 1, "é": 2}, merges=[]))
    spec.normalizer = normalizers.Lowercase()
    spec.add_tokens([AddedToken("AB", special=special, normalized=normalized)])
    tokenizer = spec.compile()
    text = "éABABAB"

    assert tokenizer.encode(text).ids == [2, 3, 3, 3]
    assert tokenizer.encode(text, untrusted_spans=[(2, 4), (6, 8)]).ids == [2, 0, 1, 3, 0, 1]
    with pytest.raises(ValueError):
        tokenizer.encode(text, untrusted_spans=[(1, 4)])


def test_encode_untrusted_spans_rejects_control_id_from_model():
    spec = Tokenizer(models.BPE(vocab={"a": 0, "b": 1, "ab": 2}, merges=[("a", "b")]))
    spec.add_tokens([AddedToken("ab", special=True)])
    tokenizer = spec.compile()

    assert tokenizer.encode("ab", encode_special_tokens=True).ids == [2]
    with pytest.raises(ValueError):
        tokenizer.encode("ab", untrusted_spans=[(0, 2)])
