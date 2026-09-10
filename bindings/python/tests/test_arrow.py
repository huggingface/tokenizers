import ctypes
import gc
import json
import re
import struct
import weakref
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from tokenizers import Padding, Tokenizer

pa = pytest.importorskip("pyarrow")


@pytest.fixture(params=[pa.string(), pa.large_string()], ids=["string", "large_string"])
def arrow_type(request):
    return request.param


@pytest.fixture
def tokenizer_config():
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "world", "arrow", "élan", "é", "你好", "🙂"]
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": len(vocab),
                "content": "[KEEP]",
                "single_word": False,
                "lstrip": True,
                "rstrip": True,
                "normalized": False,
                "special": False,
            }
        ],
        "normalizer": {"type": "Sequence", "normalizers": [{"type": "NFC"}, {"type": "Lowercase"}]},
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
            ],
            "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
            "special_tokens": {
                "[CLS]": {"id": "[CLS]", "ids": [2], "tokens": ["[CLS]"]},
                "[SEP]": {"id": "[SEP]", "ids": [3], "tokens": ["[SEP]"]},
            },
        },
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": {token: i for i, token in enumerate(vocab)}, "unk_token": "[UNK]"},
    }


@pytest.fixture
def tokenizer(tokenizer_config, tmp_path):
    path = tmp_path / "arrow-tokenizer.json"
    path.write_text(json.dumps(tokenizer_config))
    return Tokenizer.from_file(path)


def encoding_states(encodings):
    # Compare every field exposed by the pipeline bindings.
    return [(encoding.ids, encoding.type_ids, encoding.attention_mask) for encoding in encodings]


def assert_matches_list(tokenizer, array, texts, add_special_tokens=True, **arrow_kwargs):
    list_kwargs = {"padding": arrow_kwargs["padding"]} if "padding" in arrow_kwargs else {}
    expected = tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens, **list_kwargs)
    actual = tokenizer.encode_batch_arrow(array, add_special_tokens=add_special_tokens, **arrow_kwargs)
    assert encoding_states(actual) == encoding_states(expected)
    return actual


@pytest.mark.parametrize("parallelism", ["false", "true"])
@pytest.mark.parametrize("add_special_tokens", [False, True])
def test_unicode_slices_and_added_tokens(tokenizer, arrow_type, monkeypatch, parallelism, add_special_tokens):
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", parallelism)
    texts = [
        "HELLO world",
        "Élan 你好 🙂",
        "e\u0301lan",
        "",
        "hello\0world",
        "\u2003hello\tworld\n",
        "hello   [KEEP]  world",
    ]
    # Cross a validity-bitmap byte boundary and retain nulls outside the logical slice.
    parent = pa.array([None] + ["excluded"] * 8 + texts + [None], type=arrow_type)
    array = parent.slice(9, len(texts))
    assert array.offset == 9
    assert array.null_count == 0
    actual = assert_matches_list(tokenizer, array, texts, add_special_tokens)
    assert 11 in actual[-1].ids


def test_unicode_ids(tokenizer, arrow_type):
    array = pa.array(["é 你好 🙂"], type=arrow_type)
    actual = assert_matches_list(tokenizer, array, ["é 你好 🙂"], add_special_tokens=False)
    assert actual[0].ids == [8, 9, 10]


def test_byte_level_bpe_matches_list(gpt2, arrow_type):
    texts = ["hello world é 你好 🙂 " * 3, "", "hello arrow", "e\u0301lan"]
    gpt2.padding = Padding(pad_to_multiple_of=4, pad_id=50256, pad_token="<|endoftext|>")
    array = pa.array(["excluded"] + texts, type=arrow_type).slice(1)
    assert_matches_list(gpt2, array, texts)


@pytest.mark.parametrize("direction", ["left", "right"])
@pytest.mark.parametrize("length", [None, 12], ids=["batch_longest", "fixed"])
@pytest.mark.parametrize("add_special_tokens", [False, True])
def test_padding(tokenizer, arrow_type, direction, length, add_special_tokens):
    tokenizer.padding = Padding(
        direction=direction,
        length=length,
        pad_to_multiple_of=4,
        pad_id=0,
        pad_token="[PAD]",
    )
    texts = ["hello", "hello world arrow é 你好 🙂 hello world arrow", ""]
    array = pa.array(texts, type=arrow_type)
    actual = assert_matches_list(tokenizer, array, texts, add_special_tokens)
    assert {len(encoding.ids) for encoding in actual} == {12}
    assert 0 in actual[0].attention_mask


@pytest.mark.parametrize("padding", [None, Padding(length=4, pad_id=0, pad_type_id=7)])
def test_per_call_padding_preserves_config(tokenizer, arrow_type, padding):
    configured = Padding(length=8, pad_id=0)
    tokenizer.padding = configured
    texts = ["hello", ""]
    actual = assert_matches_list(tokenizer, pa.array(texts, type=arrow_type), texts, padding=padding)
    assert [len(encoding) for encoding in actual] == ([3, 2] if padding is None else [4, 4])
    assert tokenizer.padding == configured


@pytest.mark.parametrize("sliced", [False, True])
@pytest.mark.parametrize("null_handling", [None, "error", "empty", "skip"], ids=["default", "error", "empty", "skip"])
def test_empty_batch_with_padding(tokenizer, arrow_type, sliced, null_handling):
    tokenizer.padding = Padding(pad_to_multiple_of=8)
    array = pa.array(["hello", "world"], type=arrow_type).slice(1, 0) if sliced else pa.array([], type=arrow_type)
    kwargs = {} if null_handling is None else {"null_handling": null_handling}
    assert_matches_list(tokenizer, array, [], **kwargs)


@pytest.mark.parametrize("explicit", [False, True], ids=["default", "explicit_error"])
def test_null_error_reports_index_in_slice(tokenizer, arrow_type, explicit):
    parent = pa.array(["excluded"] * 9 + ["hello", None, "world"], type=arrow_type)
    array = parent.slice(9, 3)
    kwargs = {"null_handling": "error"} if explicit else {}
    with pytest.raises(ValueError) as error:
        tokenizer.encode_batch_arrow(array, **kwargs)
    message = str(error.value)
    assert "null" in message.lower()
    assert re.search(r"\b1\b", message), message
    assert not re.search(r"\b10\b", message), message


@pytest.mark.parametrize("null_handling", ["empty", "skip"])
@pytest.mark.parametrize("add_special_tokens", [False, True])
@pytest.mark.parametrize("direction", ["left", "right"])
def test_null_handling_matches_transformed_batch(tokenizer, arrow_type, null_handling, add_special_tokens, direction):
    tokenizer.padding = Padding(direction=direction, pad_to_multiple_of=4, pad_id=0, pad_token="[PAD]")
    values = ["hello", None, "", "é 你好 🙂", None, "hello world arrow é 你好 🙂 hello world arrow"]
    array = pa.array([None] + ["excluded"] * 8 + values + [None], type=arrow_type).slice(9, len(values))
    if null_handling == "empty":
        expected_texts = ["hello", "", "", "é 你好 🙂", "", "hello world arrow é 你好 🙂 hello world arrow"]
    else:
        expected_texts = ["hello", "", "é 你好 🙂", "hello world arrow é 你好 🙂 hello world arrow"]
    producer = ArrayProducer(array)
    actual = assert_matches_list(tokenizer, producer, expected_texts, add_special_tokens, null_handling=null_handling)
    assert len(actual) == (6 if null_handling == "empty" else 4)
    assert {len(encoding.ids) for encoding in actual} == {12}
    assert producer.exports == 1


@pytest.mark.parametrize("null_handling", ["error", "empty", "skip"])
@pytest.mark.parametrize("add_special_tokens", [False, True])
def test_all_null_string_batch(tokenizer, arrow_type, null_handling, add_special_tokens):
    tokenizer.padding = Padding(length=8, pad_id=0, pad_token="[PAD]")
    array = pa.array([None, None, None], type=arrow_type)
    if null_handling == "error":
        with pytest.raises(ValueError, match=r"null.*\b0\b"):
            tokenizer.encode_batch_arrow(array, add_special_tokens=add_special_tokens, null_handling=null_handling)
    else:
        expected_texts = ["", "", ""] if null_handling == "empty" else []
        actual = assert_matches_list(tokenizer, array, expected_texts, add_special_tokens, null_handling=null_handling)
        assert len(actual) == len(expected_texts)


@pytest.mark.parametrize("null_handling", ["", "drop", "EMPTY", None, 1])
@pytest.mark.parametrize("empty", [False, True])
def test_invalid_null_handling_does_not_export(tokenizer, arrow_type, null_handling, empty):
    producer = ArrayProducer(pa.array([] if empty else ["hello"], type=arrow_type))
    expected_error = ValueError if isinstance(null_handling, str) else TypeError
    with pytest.raises(expected_error):
        tokenizer.encode_batch_arrow(producer, null_handling=null_handling)
    assert producer.exports == 0


def test_null_handling_is_keyword_only(tokenizer, arrow_type):
    producer = ArrayProducer(pa.array([None], type=arrow_type))
    with pytest.raises(TypeError):
        tokenizer.encode_batch_arrow(producer, True, "empty")
    assert producer.exports == 0


@pytest.mark.parametrize(
    "array",
    [
        pa.array([1, 2], type=pa.int64()),
        pa.array([b"hello"], type=pa.binary()),
        pa.array([["hello"]], type=pa.list_(pa.string())),
        pa.array([{"text": "hello"}], type=pa.struct([("text", pa.string())])),
        pa.array(["hello"]).dictionary_encode(),
        pa.nulls(1),
    ],
    ids=["integer", "binary", "list", "struct", "dictionary", "null_type"],
)
def test_rejects_unsupported_arrow_types(tokenizer, array):
    with pytest.raises(TypeError):
        tokenizer.encode_batch_arrow(array)


@pytest.mark.parametrize(
    "input",
    [
        ["hello", "world"],
        ("hello", "world"),
        "hello",
        pa.chunked_array([["hello"], ["world"]], type=pa.string()),
    ],
    ids=["list", "tuple", "string", "chunked_array"],
)
def test_requires_array_protocol(tokenizer, input):
    with pytest.raises(TypeError):
        tokenizer.encode_batch_arrow(input)


class ArrayProducer:
    def __init__(self, array):
        self.array = array
        self.exports = 0

    def __arrow_c_array__(self, requested_schema=None):
        self.exports += 1
        return self.array.__arrow_c_array__(requested_schema)

    def __iter__(self):
        raise AssertionError("Arrow input must not be iterated through Python")

    def to_pylist(self):
        raise AssertionError("Arrow input must not be converted to a Python list")

    def to_numpy(self, *args, **kwargs):
        raise AssertionError("Arrow input must not be converted to NumPy")


def test_protocol_producer_can_export_repeatedly(tokenizer, arrow_type):
    texts = ["hello world", "é 你好 🙂", ""]
    producer = ArrayProducer(pa.array(texts, type=arrow_type))
    for _ in range(3):
        assert_matches_list(tokenizer, producer, texts)
    assert producer.exports == 3


def test_capsules_retain_temporary_array_buffers(tokenizer, arrow_type, monkeypatch):
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")
    texts = ["hello world", "é 你好 🙂", "hello [KEEP] arrow", ""] * 32

    class TemporaryArrayProducer:
        def __arrow_c_array__(self, requested_schema=None):
            array = pa.array(texts, type=arrow_type)
            capsules = array.__arrow_c_array__(requested_schema)
            del array
            gc.collect()
            return capsules

        def __iter__(self):
            raise AssertionError("Arrow input must not be iterated through Python")

    actual = assert_matches_list(tokenizer, TemporaryArrayProducer(), texts)
    gc.collect()
    assert encoding_states(actual) == encoding_states(tokenizer.encode_batch(texts))


class CapsuleProducer:
    def __init__(self, capsules):
        self.capsules = capsules

    def __arrow_c_array__(self, requested_schema=None):
        return self.capsules


def test_rejects_reused_capsules(tokenizer, arrow_type):
    texts = ["hello", "é 你好 🙂"]
    producer = CapsuleProducer(pa.array(texts, type=arrow_type).__arrow_c_array__())
    assert_matches_list(tokenizer, producer, texts)
    with pytest.raises(ValueError, match="consumed"):
        tokenizer.encode_batch_arrow(producer)


@pytest.mark.parametrize("failure", ["null", "unsupported"])
def test_rejected_batch_consumes_capsules_once(tokenizer, arrow_type, failure):
    array = pa.array([None], type=arrow_type) if failure == "null" else pa.array([1], type=pa.int64())
    error_type = ValueError if failure == "null" else TypeError
    producer = CapsuleProducer(array.__arrow_c_array__())
    with pytest.raises(error_type):
        tokenizer.encode_batch_arrow(producer)
    with pytest.raises(ValueError, match="consumed"):
        tokenizer.encode_batch_arrow(producer)


def test_shared_capsules_are_consumed_once_across_threads(tokenizer, arrow_type):
    texts = ["hello world", "é 你好 🙂"] * 32
    expected = encoding_states(tokenizer.encode_batch(texts))
    barrier = Barrier(2)

    class SharedCapsuleProducer(CapsuleProducer):
        def __arrow_c_array__(self, requested_schema=None):
            barrier.wait(timeout=10)
            return self.capsules

    producer = SharedCapsuleProducer(pa.array(texts, type=arrow_type).__arrow_c_array__())

    def encode():
        try:
            return encoding_states(tokenizer.encode_batch_arrow(producer))
        except ValueError as error:
            assert "consumed" in str(error)
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: encode(), range(2)))
    assert results.count(None) == 1
    assert next(result for result in results if result is not None) == expected


@pytest.mark.parametrize("malformed", [None, (), (None, None)], ids=["none", "empty_tuple", "non_capsules"])
def test_rejects_malformed_protocol_result(tokenizer, malformed):
    with pytest.raises((TypeError, ValueError)):
        tokenizer.encode_batch_arrow(CapsuleProducer(malformed))


def test_rejects_swapped_capsules_without_consuming_them(tokenizer, arrow_type):
    texts = ["hello world"]
    capsules = pa.array(texts, type=arrow_type).__arrow_c_array__()
    with pytest.raises(ValueError):
        tokenizer.encode_batch_arrow(CapsuleProducer(capsules[::-1]))
    assert_matches_list(tokenizer, CapsuleProducer(capsules), texts)


def test_rejects_invalid_utf8_before_tokenization(tokenizer, arrow_type):
    offset_format = "=qq" if pa.types.is_large_string(arrow_type) else "=ii"
    array = pa.Array.from_buffers(
        arrow_type,
        1,
        [None, pa.py_buffer(struct.pack(offset_format, 0, 1)), pa.py_buffer(b"\xff")],
    )
    with pytest.raises(ValueError, match="invalid Arrow array"):
        tokenizer.encode_batch_arrow(array)


@pytest.mark.parametrize("null_handling", ["error", "empty", "skip"])
def test_null_payload_does_not_require_valid_utf8(tokenizer, arrow_type, null_handling):
    offset_format = "=qqqq" if pa.types.is_large_string(arrow_type) else "=iiii"
    array = pa.Array.from_buffers(
        arrow_type,
        3,
        [
            pa.py_buffer(b"\x05"),
            pa.py_buffer(struct.pack(offset_format, 0, 5, 6, 8)),
            pa.py_buffer(b"hello\xff\xc3\xa9"),
        ],
    )
    producer = ArrayProducer(array)
    if null_handling == "error":
        with pytest.raises(ValueError, match=r"null.*\b1\b"):
            tokenizer.encode_batch_arrow(producer, null_handling=null_handling)
    else:
        expected_texts = ["hello", "", "é"] if null_handling == "empty" else ["hello", "é"]
        assert_matches_list(tokenizer, producer, expected_texts, null_handling=null_handling)
    assert producer.exports == 1


@pytest.mark.parametrize("unaligned", [False, True], ids=["aligned", "unaligned"])
def test_offsets_buffer_alignment(tokenizer, arrow_type, unaligned):
    width = 8 if pa.types.is_large_string(arrow_type) else 4
    offset_format = "=qqq" if width == 8 else "=iii"
    packed_offsets = struct.pack(offset_format, 0, 5, 7)
    prefix = b"\0" if unaligned else b""
    # Arrow's allocator supplies aligned memory; slicing by one byte deliberately
    # removes that alignment without introducing invalid pointers or buffer lengths.
    buffer = pa.allocate_buffer(len(prefix) + len(packed_offsets))
    memoryview(buffer).cast("B")[:] = prefix + packed_offsets
    offsets = buffer.slice(len(prefix))
    assert offsets.address % width == (1 if unaligned else 0)
    array = pa.Array.from_buffers(arrow_type, 2, [None, offsets, pa.py_buffer(b"hello\xc3\xa9")])
    producer = ArrayProducer(array)
    if unaligned:
        with pytest.raises(ValueError, match="aligned"):
            tokenizer.encode_batch_arrow(producer)
    else:
        assert_matches_list(tokenizer, producer, ["hello", "é"])
    assert producer.exports == 1


# These layouts are the Arrow C Data Interface structs. Every pointer below comes
# from a genuine PyArrow export; hooks retain and delegate to its release callback.
class ArrowSchema(ctypes.Structure):
    _fields_ = [
        ("format", ctypes.c_void_p),
        ("name", ctypes.c_void_p),
        ("metadata", ctypes.c_void_p),
        ("flags", ctypes.c_int64),
        ("n_children", ctypes.c_int64),
        ("children", ctypes.c_void_p),
        ("dictionary", ctypes.c_void_p),
        ("release", ctypes.c_void_p),
        ("private_data", ctypes.c_void_p),
    ]


class ArrowArray(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_int64),
        ("null_count", ctypes.c_int64),
        ("offset", ctypes.c_int64),
        ("n_buffers", ctypes.c_int64),
        ("n_children", ctypes.c_int64),
        ("buffers", ctypes.c_void_p),
        ("children", ctypes.c_void_p),
        ("dictionary", ctypes.c_void_p),
        ("release", ctypes.c_void_p),
        ("private_data", ctypes.c_void_p),
    ]


class TrackedCapsuleProducer(CapsuleProducer):
    def __init__(self, array, invalid_schema=False):
        super().__init__(array.__arrow_c_array__())
        self.releases = {"schema": 0, "array": 0}
        self._callbacks = []
        get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
        get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        get_pointer.restype = ctypes.c_void_p
        for capsule, kind, struct_type in zip(self.capsules, ("schema", "array"), (ArrowSchema, ArrowArray)):
            pointer = ctypes.cast(get_pointer(capsule, f"arrow_{kind}".encode()), ctypes.POINTER(struct_type))
            callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_type))
            original_release = callback_type(pointer.contents.release)
            original_format = pointer.contents.format if kind == "schema" else None
            releases = self.releases

            def release(pointer, kind=kind, original_release=original_release, original_format=original_format):
                releases[kind] += 1
                if kind == "schema":
                    # Restore the producer's format before it reclaims schema data.
                    pointer.contents.format = original_format
                original_release(pointer)

            callback = callback_type(release)
            self._callbacks.append(callback)
            pointer.contents.release = ctypes.cast(callback, ctypes.c_void_p).value
            if kind == "schema" and invalid_schema:
                self._format = ctypes.create_string_buffer(b"invalid_arrow_format")
                pointer.contents.format = ctypes.cast(self._format, ctypes.c_void_p).value

    def __del__(self):
        # Capsules must release before the Python callback functions are destroyed,
        # including when a test fails before the consumer takes ownership.
        self.capsules = ()


@pytest.mark.parametrize("outcome", ["success", "empty", "unsupported", "schema", "null", "utf8", "encode_error"])
def test_releases_once_and_does_not_retain_python_buffer_owner(
    tokenizer, tokenizer_config, tmp_path, arrow_type, outcome
):
    class BufferOwner(bytearray):
        pass

    owner = BufferOwner(b"\xff" if outcome == "utf8" else b"hello")
    owner_ref = weakref.ref(owner)
    offset_format = "=qq" if pa.types.is_large_string(arrow_type) else "=ii"
    data_type = arrow_type
    if outcome == "unsupported":
        data_type = pa.large_binary() if pa.types.is_large_string(arrow_type) else pa.binary()
    validity = pa.py_buffer(b"\x00") if outcome == "null" else None
    array = pa.Array.from_buffers(
        data_type,
        1,
        [validity, pa.py_buffer(struct.pack(offset_format, 0, len(owner))), pa.py_buffer(owner)],
    )
    if outcome == "empty":
        array = array.slice(1, 0)
    producer = TrackedCapsuleProducer(array, invalid_schema=outcome == "schema")
    releases = producer.releases
    producer_ref = weakref.ref(producer)
    del array, owner
    gc.collect()
    assert owner_ref() is not None

    if outcome == "encode_error":
        tokenizer_config["model"]["vocab"] = {"world": 0}
        tokenizer_config["added_tokens"] = []
        path = tmp_path / "missing-unk.json"
        path.write_text(json.dumps(tokenizer_config))
        tokenizer = Tokenizer.from_file(path)

    if outcome in {"success", "empty"}:
        actual = tokenizer.encode_batch_arrow(producer)
        expected_texts = ["hello"] if outcome == "success" else []
        assert encoding_states(actual) == encoding_states(tokenizer.encode_batch(expected_texts))
    else:
        expected_error = TypeError if outcome == "unsupported" else ValueError
        message = {
            "unsupported": "expected an Arrow string",
            "schema": "invalid Arrow schema",
            "null": r"null.*\b0\b",
            "utf8": "invalid UTF-8",
            "encode_error": "Missing.*UNK",
        }[outcome]
        with pytest.raises(expected_error, match=message):
            tokenizer.encode_batch_arrow(producer)

    gc.collect()
    assert releases == {"schema": 1, "array": 1}
    assert owner_ref() is None
    del producer
    gc.collect()
    assert producer_ref() is None
    assert releases == {"schema": 1, "array": 1}
