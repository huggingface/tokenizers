import pytest

from tokenizers import Tokenizer, Truncation

# Every id in this file comes from released tokenizers 0.23.1 on the same fixture
LONG = "Hello there, how are you today?"
GPT2_IDS = [15496, 612, 11, 703, 389, 345, 1909, 30]
WIKI_IDS = [27253, 5503, 16, 6447, 5112, 6218, 8773, 35]


def test_no_truncation(gpt2):
    assert gpt2.truncation is None
    assert gpt2.encode(LONG).ids == GPT2_IDS


def test_max_length(gpt2):
    gpt2.truncation = Truncation(4)

    assert gpt2.encode(LONG).ids == GPT2_IDS[:4]


def test_shorter_than_max_length(gpt2):
    gpt2.truncation = Truncation(4)

    assert gpt2.encode("Hello there").ids == [15496, 612]


def test_left(gpt2):
    gpt2.truncation = Truncation(4, direction="left")

    assert gpt2.encode(LONG).ids == GPT2_IDS[-4:]


def test_specials_count_toward_max_length(bert):
    bert.truncation = Truncation(5)

    encoding = bert.encode(LONG)

    assert encoding.ids == [1, 27462, 7495, 16, 2]
    assert encoding.attention_mask == [1] * 5
    assert bert.encode(LONG, add_special_tokens=False).ids == [27462, 7495, 16, 7510, 7268]


def test_specials_wrap_what_left_truncation_kept(bert):
    bert.truncation = Truncation(5, direction="left")

    assert bert.encode(LONG).ids == [1, 7989, 9819, 35, 2]


def test_only_first(gpt2):
    gpt2.truncation = Truncation(4, strategy="only_first")

    assert gpt2.encode(LONG).ids == GPT2_IDS[:4]


# Tokenizer.encode takes one sequence, so only_second never has the sequence it cuts.
def test_only_second_has_nothing_to_cut(gpt2):
    gpt2.truncation = Truncation(4, strategy="only_second")

    with pytest.raises(ValueError, match="Second sequence not provided"):
        gpt2.encode(LONG)


def test_unknown_strategy():
    with pytest.raises(ValueError, match='strategy must be "longest_first", "only_first" or "only_second"'):
        Truncation(4, strategy="sideways")  # ty: ignore[invalid-argument-type]


def test_unknown_direction():
    with pytest.raises(ValueError, match='direction must be "left" or "right"'):
        Truncation(4, direction="sideways")  # ty: ignore[invalid-argument-type]


def test_max_length_is_required():
    with pytest.raises(TypeError):
        Truncation()  # ty: ignore[missing-argument]


def test_truncation_from_file(truncated_wiki):
    tokenizer = Tokenizer.from_file(truncated_wiki)

    assert tokenizer.truncation == Truncation(4)
    assert tokenizer.encode(LONG).ids == WIKI_IDS[:4]


def test_encode_truncation_override(gpt2):
    assert gpt2.encode(LONG, truncation=Truncation(4)).ids == GPT2_IDS[:4]
    assert gpt2.truncation is None


def test_encode_truncation_off(gpt2):
    gpt2.truncation = Truncation(4)

    assert gpt2.encode(LONG, truncation=None).ids == GPT2_IDS
    assert gpt2.encode(LONG).ids == GPT2_IDS[:4]


def test_encode_truncation_override_replaces_config(gpt2):
    gpt2.truncation = Truncation(4)

    assert gpt2.encode(LONG, truncation=Truncation(2)).ids == GPT2_IDS[:2]
    assert gpt2.encode(LONG).ids == GPT2_IDS[:4]


def test_encode_batch_truncation_override(gpt2):
    short, long = gpt2.encode_batch(["Hello", LONG], truncation=Truncation(4))

    assert short.ids == [15496]
    assert long.ids == GPT2_IDS[:4]
    assert gpt2.truncation is None


def test_encode_batch_truncation_off(gpt2):
    gpt2.truncation = Truncation(4)

    assert [e.ids for e in gpt2.encode_batch(["Hello", LONG], truncation=None)] == [[15496], GPT2_IDS]
    assert [e.ids for e in gpt2.encode_batch(["Hello", LONG])] == [[15496], GPT2_IDS[:4]]


def test_set_truncation(truncated_wiki):
    tokenizer = Tokenizer.from_file(truncated_wiki)

    tokenizer.truncation = Truncation(2)
    assert tokenizer.truncation == Truncation(2)
    assert tokenizer.encode(LONG).ids == WIKI_IDS[:2]

    tokenizer.truncation = None
    assert tokenizer.truncation is None
    assert tokenizer.encode(LONG).ids == WIKI_IDS


def test_defaults():
    truncation = Truncation(512)

    assert truncation.max_length == 512
    assert truncation.strategy == "longest_first"
    assert truncation.direction == "right"
