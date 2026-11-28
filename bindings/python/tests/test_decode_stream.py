import pytest

from tokenizers.decoders import DecodeStream

TEXT = "Hello, streaming 世界"


def stream(tokenizer, decode_stream, ids):
    return "".join(decode_stream.step(tokenizer, id) or "" for id in ids)


def test_steps_add_up_to_decode(gpt2):
    ids = gpt2.encode(TEXT).ids

    assert stream(gpt2, DecodeStream(), ids) == gpt2.decode(ids)


def test_step_takes_a_list_of_ids(gpt2):
    ids = gpt2.encode(TEXT).ids

    assert DecodeStream().step(gpt2, ids) == gpt2.decode(ids)


def test_prompt_ids_are_not_returned(gpt2):
    prompt = gpt2.encode("Hello there,").ids
    generated = gpt2.encode(" general Kenobi").ids

    assert stream(gpt2, DecodeStream(ids=prompt), generated) == " general Kenobi"


def test_flush_returns_the_held_back_text(gpt2):
    # "世" takes two byte-level tokens; the stream stops after the first one
    ids = gpt2.encode("Hello 世").ids
    decode_stream = DecodeStream()
    text = stream(gpt2, decode_stream, ids[:-1])

    assert text + decode_stream.flush(gpt2) == gpt2.decode(ids[:-1])
    # flush does not consume the held-back bytes: the next step still completes "世"
    assert text + decode_stream.step(gpt2, ids[-1]) == gpt2.decode(ids)
    assert decode_stream.flush(gpt2) is None


def test_flush_does_not_return_the_prompt(gpt2):
    decode_stream = DecodeStream(ids=gpt2.encode("Hello there,").ids)

    assert decode_stream.flush(gpt2) is None


def test_keeps_special_tokens_by_default(llama3):
    ids = llama3.encode("Hello").ids

    assert stream(llama3, DecodeStream(), ids) == llama3.decode(ids, skip_special_tokens=False)
    assert stream(llama3, DecodeStream(skip_special_tokens=True), ids) == llama3.decode(ids)


# vLLM catches TypeError and OverflowError, and skips the invalid id.
@pytest.mark.parametrize("id", [-1, 2**32, "1"])
def test_an_invalid_id_is_a_type_error(gpt2, id):
    with pytest.raises(TypeError):
        DecodeStream().step(gpt2, id)
