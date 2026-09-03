import gc
import importlib.metadata
import json
import math
import multiprocessing
import pickle
import sys
import sysconfig
import threading
from pathlib import Path

import numpy as np
import pytest

import tokenizers
from tokenizers import Encoding, Padding, Tokenizer, __version__

# `make test` fetches these; see the Makefile's TESTS_RESOURCES.
DATA = Path(__file__).parent.parent / "data"
# BPE with a ByteLevel pre-tokenizer, post-processor and decoder.
GPT2 = DATA / "gpt2.json"
# BPE with a Whitespace pre-tokenizer and nothing else.
WIKI = DATA / "tokenizer-wiki.json"
# WordPiece with a `[CLS] $A [SEP]` template.
BERT = DATA / "bert-wiki.json"

SHORT_AND_LONG = ["Hello", "Hello there, how are you today?"]


@pytest.fixture
def padded_wiki(tmp_path):
    declared = json.loads(WIKI.read_text())
    declared["padding"] = {
        "strategy": "BatchLongest",
        "direction": "Right",
        "pad_to_multiple_of": None,
        "pad_id": 0,
        "pad_type_id": 0,
        "pad_token": "[PAD]",
    }
    path = tmp_path / "padded.json"
    path.write_text(json.dumps(declared))
    return path


def test_module_reports_the_installed_version():
    assert __version__ == importlib.metadata.version("tokenizers")


def test_classes_live_in_the_tokenizers_module():
    assert {cls.__module__ for cls in (Tokenizer, Padding, Encoding)} == {"tokenizers"}


def test_public_api_is_documented():
    assert tokenizers.__doc__
    for cls in (Tokenizer, Padding, Encoding):
        assert cls.__doc__, cls
        for name in dir(cls):
            if not name.startswith("_"):
                assert getattr(cls, name).__doc__, f"{cls.__name__}.{name}"


def test_from_file_reads_a_legacy_tokenizer_json():
    tokenizer = Tokenizer.from_file(WIKI)
    assert isinstance(tokenizer, Tokenizer)


def test_from_file_accepts_a_str_path():
    assert isinstance(Tokenizer.from_file(str(WIKI)), Tokenizer)


def test_from_file_reports_a_missing_file():
    with pytest.raises(FileNotFoundError, match="does-not-exist.json"):
        Tokenizer.from_file(DATA / "does-not-exist.json")


def test_tokenizer_repr_shows_the_pipeline():
    shown = repr(Tokenizer.from_file(GPT2))

    assert shown.startswith(
        'Tokenizer(version="2.0", truncation=None, padding=None, role_to_token=None, added_tokens=[{'
    )
    # The pipeline rebuilds GPT-2's ByteLevel pre-tokenizer as a Split plus a byte-level model.
    for stage in [
        "normalizer=None",
        'pre_tokenizer=Split(behavior="Isolated", invert=False, pattern={"Regex": ',
        "post_processor=None",
        "decoder=ByteLevel()",
        "model=BPE(byte_fallback=False, byte_level=True, ",
    ]:
        assert f", {stage}" in shown
    # The vocab and the merges are cut after a few entries, not dumped.
    assert ", ...}" in shown and ", ...]" in shown
    assert len(shown) < 2000


def test_tokenizer_repr_shows_the_padding_in_force_not_the_files(padded_wiki):
    tokenizer = Tokenizer.from_file(padded_wiki)

    tokenizer.padding = Padding(length=8)

    assert f", padding={Padding(length=8)!r}, " in repr(tokenizer)


def test_encode_returns_an_encoding():
    encoding = Tokenizer.from_file(BERT).encode("Hello there")

    assert isinstance(encoding, Encoding)
    assert encoding.ids[0] == 1 and encoding.ids[-1] == 2
    assert len(encoding) == len(encoding.ids)
    assert encoding.type_ids.tolist() == [0] * len(encoding)
    assert encoding.attention_mask.tolist() == [1] * len(encoding)


def test_encodings_with_the_same_fields_are_equal():
    tokenizer = Tokenizer.from_file(BERT)

    assert tokenizer.encode("Hello") == tokenizer.encode("Hello")
    assert tokenizer.encode("Hello") != tokenizer.encode("Goodbye")


def test_encoding_repr_shows_its_fields():
    encoding = Tokenizer.from_file(BERT).encode("Hi", add_special_tokens=False)

    assert repr(encoding) == f"Encoding(ids={encoding.ids.tolist()}, type_ids=[0], attention_mask=[1])"


def test_add_special_tokens_is_honoured():
    tokenizer = Tokenizer.from_file(BERT)

    with_specials = tokenizer.encode("Hello", add_special_tokens=True)
    without = tokenizer.encode("Hello", add_special_tokens=False)

    assert len(with_specials) == len(without) + 2


def test_encode_batch_keeps_input_order():
    tokenizer = Tokenizer.from_file(GPT2)
    texts = ["Hello there", "General Kenobi", "You are a bold one"]

    batch = tokenizer.encode_batch(texts)

    assert [e.ids.tolist() for e in batch] == [tokenizer.encode(t).ids.tolist() for t in texts]


def test_encode_batch_of_nothing_is_empty():
    assert Tokenizer.from_file(GPT2).encode_batch([]) == []


def test_decode_round_trips_through_gpt2():
    tokenizer = Tokenizer.from_file(GPT2)
    text = "Hello there, how are you?"

    assert tokenizer.decode(tokenizer.encode(text).ids) == text


def test_decode_skips_special_tokens_by_default():
    # This BERT lowercases, so the round trip lands on the normalized text.
    tokenizer = Tokenizer.from_file(BERT)
    ids = tokenizer.encode("Hello there").ids

    assert tokenizer.decode(ids) == "hello there"
    assert tokenizer.decode(ids, skip_special_tokens=False) == "[CLS] hello there [SEP]"


def test_padding_defaults_match_the_released_enable_padding():
    padding = Padding()

    assert padding.direction == "right"
    assert padding.pad_id == 0
    assert padding.pad_type_id == 0
    assert padding.pad_token == "[PAD]"
    assert padding.length is None
    assert padding.pad_to_multiple_of is None


def test_padding_refuses_an_unknown_direction():
    with pytest.raises(ValueError, match="direction"):
        Padding(direction="up")  # ty: ignore[invalid-argument-type]


def test_paddings_with_the_same_parameters_are_equal():
    assert Padding(length=8, pad_id=3) == Padding(length=8, pad_id=3)
    assert hash(Padding(length=8, pad_id=3)) == hash(Padding(length=8, pad_id=3))
    assert Padding(length=8) != Padding()
    assert Padding(direction="left") != Padding()


def test_padding_repr_round_trips():
    padding = Padding(direction="left", pad_id=50256, pad_token="<|endoftext|>", pad_to_multiple_of=8)

    assert eval(repr(padding)) == padding


def test_no_padding_unless_asked():
    tokenizer = Tokenizer.from_file(GPT2)
    assert tokenizer.padding is None

    short, long = tokenizer.encode_batch(SHORT_AND_LONG)

    assert len(short) < len(long)


def test_padding_to_the_longest_in_the_batch():
    padding = Padding(direction="left", pad_id=50256, pad_token="<|endoftext|>")
    tokenizer = Tokenizer.from_file(GPT2, padding=padding)

    in_force = tokenizer.padding
    assert in_force is not None
    assert in_force.direction == "left"
    assert in_force.pad_id == 50256
    short, long = tokenizer.encode_batch(SHORT_AND_LONG)

    assert len(short) == len(long)
    pad = len(long) - 1
    assert short.ids[:pad].tolist() == [50256] * pad
    assert short.attention_mask.tolist() == [0] * pad + [1]
    assert long.attention_mask.tolist() == [1] * len(long)


def test_padding_to_a_fixed_length():
    tokenizer = Tokenizer.from_file(GPT2, padding=Padding(length=8, pad_id=50256))

    encoding = tokenizer.encode("Hello")

    assert len(encoding) == 8
    assert encoding.ids[1:].tolist() == [50256] * 7
    assert encoding.attention_mask.tolist() == [1] + [0] * 7


def test_padding_to_a_multiple():
    unpadded = Tokenizer.from_file(GPT2)
    padded = Tokenizer.from_file(GPT2, padding=Padding(pad_to_multiple_of=8))

    assert len(padded.encode("Hello")) == 8
    for text in ["Hello there, how are you today?", "word " * 20]:
        assert len(padded.encode(text)) == math.ceil(len(unpadded.encode(text)) / 8) * 8


def test_padding_declared_in_the_file_is_applied(padded_wiki):
    tokenizer = Tokenizer.from_file(padded_wiki)

    in_force = tokenizer.padding
    assert in_force is not None
    assert in_force.direction == "right"
    assert in_force.length is None
    short, long = tokenizer.encode_batch(SHORT_AND_LONG)
    assert len(short) == len(long)
    assert short.ids[-1] == 0 and short.attention_mask[-1] == 0


def test_padding_passed_to_from_file_replaces_the_files(padded_wiki):
    tokenizer = Tokenizer.from_file(padded_wiki, padding=Padding(direction="left"))

    in_force = tokenizer.padding
    assert in_force is not None
    assert in_force.direction == "left"
    short, _ = tokenizer.encode_batch(SHORT_AND_LONG)
    assert short.ids[0] == 0 and short.attention_mask[0] == 0


def test_padding_can_be_set_after_construction():
    tokenizer = Tokenizer.from_file(GPT2)

    tokenizer.padding = Padding(pad_id=50256)

    in_force = tokenizer.padding
    assert in_force is not None
    assert in_force.pad_id == 50256
    short, long = tokenizer.encode_batch(SHORT_AND_LONG)
    assert len(short) == len(long)
    assert short.ids[-1] == 50256


def test_setting_padding_to_none_switches_it_off(padded_wiki):
    tokenizer = Tokenizer.from_file(padded_wiki)

    tokenizer.padding = None

    assert tokenizer.padding is None
    short, long = tokenizer.encode_batch(SHORT_AND_LONG)
    assert len(short) < len(long)


def test_padding_can_change_while_another_thread_encodes():
    tokenizer = Tokenizer.from_file(GPT2)
    texts = [GPT2.read_text()[:4000]] * 200
    failures = []

    def encode_repeatedly():
        try:
            for _ in range(20):
                tokenizer.encode_batch(texts)
        except Exception as e:
            failures.append(e)

    encoder = threading.Thread(target=encode_repeatedly)
    encoder.start()
    while encoder.is_alive():
        tokenizer.padding = Padding(pad_id=50256)
        tokenizer.padding = None
    encoder.join()

    assert failures == []


def test_padding_and_encoding_are_read_only():
    padding = Padding()
    encoding = Tokenizer.from_file(GPT2).encode("Hello")

    with pytest.raises(AttributeError):
        padding.pad_id = 1  # ty: ignore[invalid-assignment]
    with pytest.raises(AttributeError):
        encoding.ids = []  # ty: ignore[invalid-assignment]


def test_encoding_fields_are_read_only_numpy_arrays():
    encoding = Tokenizer.from_file(BERT).encode("Hello there")

    for field in (encoding.ids, encoding.type_ids, encoding.attention_mask):
        assert isinstance(field, np.ndarray)
        assert field.dtype == np.uint32
        assert field.shape == (len(encoding),)
        assert not field.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        encoding.ids[0] = 0


def test_encoding_fields_are_views_not_copies():
    encoding = Tokenizer.from_file(BERT).encode("Hello there")

    assert np.shares_memory(encoding.ids, encoding.ids)
    assert not np.shares_memory(encoding.ids, encoding.type_ids)
    assert encoding.ids.base is encoding


def test_encoding_field_keeps_the_encoding_alive():
    ids = Tokenizer.from_file(BERT).encode("Hello there").ids
    expected = ids.tolist()

    gc.collect()

    assert isinstance(ids.base, Encoding)
    assert ids.tolist() == expected


def test_decode_takes_any_integer_sequence():
    tokenizer = Tokenizer.from_file(BERT)
    ids = tokenizer.encode("Hello there").ids

    assert tokenizer.decode(ids) == "hello there"
    assert tokenizer.decode(ids.tolist()) == "hello there"
    assert tokenizer.decode(tuple(ids.tolist())) == "hello there"
    assert tokenizer.decode(ids.astype(np.int64)) == "hello there"
    assert tokenizer.decode(ids[::2]) == tokenizer.decode(ids[::2].tolist())


def test_decode_refuses_floats():
    tokenizer = Tokenizer.from_file(BERT)
    ids = tokenizer.encode("Hello there").ids

    with pytest.raises(TypeError):
        tokenizer.decode(ids.astype(np.float64))  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError):
        tokenizer.decode([1.5, 2.5])  # ty: ignore[invalid-argument-type]


def test_tokenizer_can_be_pickled():
    tokenizer = Tokenizer.from_file(BERT)

    restored = pickle.loads(pickle.dumps(tokenizer))

    assert restored.encode("Hello there").ids.tolist() == tokenizer.encode("Hello there").ids.tolist()


def _encode_ids(tokenizer, text):
    return tokenizer.encode(text).ids.tolist()


def test_tokenizer_can_be_used_from_a_multiprocessing_worker():
    # `spawn` pickles the tokenizer to hand it to the worker process; PyTorch's DataLoader and
    # `datasets.map(num_proc=...)` do the same.
    tokenizer = Tokenizer.from_file(BERT)

    with multiprocessing.get_context("spawn").Pool(1) as pool:
        ids = pool.apply(_encode_ids, (tokenizer, "Hello there"))

    assert ids == tokenizer.encode("Hello there").ids.tolist()


# `sys._is_gil_enabled` exists on every build from 3.13 on, so only the config var tells the two apart.
@pytest.mark.skipif(not sysconfig.get_config_var("Py_GIL_DISABLED"), reason="only meaningful on a free-threaded build")
def test_importing_the_extension_keeps_the_gil_disabled():
    assert sys._is_gil_enabled() is False  # ty: ignore[unresolved-attribute]
