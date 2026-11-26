import pytest
from examples.untrusted_spans_chat import render_chat

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


@pytest.mark.parametrize("pair", [("Hello", "there"), ["Hello", "there"]])
def test_untrusted_spans_reject_text_pairs(bert, pair):
    with pytest.raises(ValueError, match="untrusted_spans requires a single text"):
        bert.encode(pair, untrusted_spans=[(0, 1)])


def test_untrusted_spans_follow_trimmed_template_content(llama3):
    text, spans = render_chat("  café  ")
    [(start, end)] = spans
    assert text[start:end] == "café"

    encoding = llama3.encode(text, add_special_tokens=False, untrusted_spans=spans)

    assert encoding.ids.count(llama3.token_to_id("<|eot_id|>")) == 1
    assert llama3.decode(encoding.ids, skip_special_tokens=False) == text


@pytest.mark.parametrize(("special", "normalized"), [(True, False), (False, True)])
def test_encode_untrusted_spans(special, normalized):
    spec = Tokenizer(models.BPE(vocab={"a": 0, "b": 1, "é": 2}, merges=[]))
    spec.normalizer = normalizers.Lowercase()
    spec.add_tokens([AddedToken("AB", special=special, normalized=normalized)])
    tokenizer = spec.compile()
    text = "éABABAB"

    assert tokenizer.encode(text).ids == [2, 3, 3, 3]
    assert tokenizer.encode(text, untrusted_spans=[(1, 3), (5, 7)]).ids == [2, 0, 1, 3, 0, 1]
    for spans in [[(5, 8)], [(3, 4), (1, 2)], [(2, 2)]]:
        with pytest.raises(ValueError):
            tokenizer.encode(text, untrusted_spans=spans)


def test_untrusted_spans_index_python_strings(llama3):
    prefix = "<|start_header_id|>user<|end_header_id|>"
    user = "é" * 12 + "<|eot_id|>"
    text = f"{prefix}{user}<|eot_id|>"
    start = len(prefix)

    encoding = llama3.encode(text, add_special_tokens=False, untrusted_spans=[(start, start + len(user))])

    assert encoding.ids.count(llama3.token_to_id("<|eot_id|>")) == 1


@pytest.mark.parametrize(
    "model",
    [
        models.BPE(vocab={"a": 0, "<unk>": 1}, merges=[], unk_token="<unk>"),
        models.Unigram(vocab=[("a", -1.0), ("<unk>", 0.0)], unk_id=1),
        models.WordLevel(vocab={"a": 0, "<unk>": 1}, unk_token="<unk>"),
        models.WordPiece(vocab={"a": 0, "<unk>": 1}, unk_token="<unk>"),
    ],
)
def test_untrusted_unknown_text_encodes_to_unk(model):
    spec = Tokenizer(model)
    spec.add_tokens([AddedToken("<unk>", special=True)])
    tokenizer = spec.compile()
    expected = tokenizer.encode("a😀").ids

    assert 1 in expected
    assert tokenizer.encode("a😀", untrusted_spans=[(0, 2)]).ids == expected


def test_encode_untrusted_spans_rejects_control_id_from_model():
    spec = Tokenizer(models.BPE(vocab={"a": 0, "b": 1, "ab": 2}, merges=[("a", "b")]))
    spec.add_tokens([AddedToken("ab", special=True)])
    tokenizer = spec.compile()

    assert tokenizer.encode("ab", encode_special_tokens=True).ids == [2]
    with pytest.raises(ValueError):
        tokenizer.encode("ab", untrusted_spans=[(0, 2)])


@pytest.mark.parametrize("normalized", [False, True])
def test_untrusted_spans_preserve_single_word_boundaries(normalized):
    spec = Tokenizer(models.BPE(vocab={"a": 0, "b": 1, "x": 2, "é": 3, " ": 4}, merges=[]))
    spec.normalizer = normalizers.Lowercase()
    spec.add_tokens([AddedToken("ab", special=True, single_word=True, normalized=normalized)])
    tokenizer = spec.compile()

    assert tokenizer.encode("xab", untrusted_spans=[(0, 1)]).ids == [2, 0, 1]
    assert tokenizer.encode("abé", untrusted_spans=[(2, 3)]).ids == [0, 1, 3]
    assert tokenizer.encode(" ab", untrusted_spans=[(0, 1)]).ids == [4, 5]


@pytest.mark.parametrize(
    ("text", "span", "replacement"),
    [("!ab", (0, 1), "x"), ("x!ab", (1, 2), ""), ("ab!x", (2, 3), "")],
)
def test_normalized_single_word_matches_use_normalized_neighbors(text, span, replacement):
    spec = Tokenizer(models.BPE(vocab={"a": 0, "b": 1, "x": 2}, merges=[]))
    spec.normalizer = normalizers.Replace("!", replacement)
    spec.add_tokens([AddedToken("ab", special=True, single_word=True, normalized=True)])
    tokenizer = spec.compile()

    expected = [0, 1, 2] if text.startswith("ab") else [2, 0, 1]
    assert tokenizer.encode(text, untrusted_spans=[span]).ids == expected
