import asyncio

import numpy as np
import pytest

from .conftest import SENTENCES, train_word_tokenizer


def test_async_encode_matches_sync():
    tok = train_word_tokenizer()

    async def go():
        single = await tok.async_encode(SENTENCES[0], add_special_tokens=False)
        batch = await tok.async_encode_batch(SENTENCES, add_special_tokens=False)
        return single, batch

    single, batch = asyncio.run(go())
    assert np.array_equal(single, tok.encode(SENTENCES[0], add_special_tokens=False))
    for got, want in zip(batch, tok.encode_batch(SENTENCES, add_special_tokens=False)):
        assert np.array_equal(got, want)


def test_async_encodes_overlap():
    tok = train_word_tokenizer()

    async def go():
        return await asyncio.gather(*(tok.async_encode(s, add_special_tokens=False) for s in SENTENCES))

    results = asyncio.run(go())
    for got, line in zip(results, SENTENCES):
        assert np.array_equal(got, tok.encode(line, add_special_tokens=False))


def test_async_error_surfaces_at_await():
    tok = train_word_tokenizer()

    async def go():
        await tok.async_encode(123, add_special_tokens=False)

    with pytest.raises(TypeError):
        asyncio.run(go())
