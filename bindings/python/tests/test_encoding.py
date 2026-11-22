import pytest

from tokenizers import Padding

FIELDS = ("ids", "type_ids", "attention_mask")


def test_torch_reads_each_field_as_int64(bert):
    torch = pytest.importorskip("torch")
    encoding = bert.encode("Hello there", padding=Padding(length=8, pad_id=3))

    for field in FIELDS:
        tensor = torch.tensor(getattr(encoding, f"{field}_dlpack"))
        assert tensor.dtype == torch.int64
        assert tensor.tolist() == getattr(encoding, field)


def test_numpy_reads_each_field_as_int64(bert):
    np = pytest.importorskip("numpy")
    encoding = bert.encode("Hello there", padding=Padding(length=8, pad_id=3))

    for field in FIELDS:
        array = np.from_dlpack(getattr(encoding, f"{field}_dlpack"))
        assert array.dtype == np.int64
        assert array.tolist() == getattr(encoding, field)


def test_dlpack_refuses_another_device(bert):
    cuda = (2, 0)

    # The export is a new array in host memory. The reader moves it to another device.
    with pytest.raises(BufferError):
        bert.encode("Hello there").ids_dlpack.__dlpack__(dl_device=cuda)


def test_dlpack_refuses_copy_false(bert):
    # The export widens to int64, so it cannot share the encoding's uint32 buffer.
    with pytest.raises(BufferError):
        bert.encode("Hello there").ids_dlpack.__dlpack__(copy=False)


def test_empty_encoding_exports_an_empty_tensor(gpt2):
    torch = pytest.importorskip("torch")
    ids = torch.tensor(gpt2.encode("").ids_dlpack)

    assert ids.shape == (0,)
    assert ids.dtype == torch.int64


def test_equality_ignores_which_fields_were_read(bert):
    read, unread = bert.encode("Hello there"), bert.encode("Hello there")
    read.type_ids, read.attention_mask

    assert read == unread


def test_equality_compares_tokens(bert):
    assert bert.encode("Hello there") != bert.encode("General Kenobi")
