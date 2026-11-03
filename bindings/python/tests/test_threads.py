import multiprocessing
import sys
import sysconfig
import threading

import pytest

import tokenizers
from conftest import GPT2


def test_concurrent_encode(gpt2):
    texts = [f"Sentence number {n}. " * (n % 17 + 1) for n in range(300)]
    expected = [e.ids for e in gpt2.encode_batch(texts)]
    results = []

    def encode():
        for _ in range(5):
            results.append([e.ids for e in gpt2.encode_batch(texts)])

    threads = [threading.Thread(target=encode) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 40
    assert all(run == expected for run in results)


def test_read_array_while_decoding_it(gpt2):
    encoding = gpt2.encode(GPT2.read_text()[:4000])
    ids = encoding.ids_array

    def decode():
        for _ in range(100):
            gpt2.decode(ids)

    decoder = threading.Thread(target=decode)
    decoder.start()
    while decoder.is_alive():
        encoding.ids_array
    decoder.join()


# `sys._is_gil_enabled` exists on every build from 3.13 on, so only the config var tells the two apart.
@pytest.mark.skipif(not sysconfig.get_config_var("Py_GIL_DISABLED"), reason="only meaningful on a free-threaded build")
def test_gil_disabled():
    assert sys._is_gil_enabled() is False  # ty: ignore[unresolved-attribute]


def test_max_threads_roundtrip():
    tokenizers.set_max_threads(3)
    try:
        assert tokenizers.get_max_threads() == 3
    finally:
        tokenizers.set_max_threads(0)
    assert tokenizers.get_max_threads() >= 1


def test_per_call_max_threads(gpt2):
    texts = [GPT2.read_text()[:20_000]] * 8
    expected = [e.ids for e in gpt2.encode_batch(texts)]
    for max_threads in (1, 2, 64):
        assert [e.ids for e in gpt2.encode_batch(texts, max_threads=max_threads)] == expected
    assert gpt2.encode(texts[0], max_threads=1).ids == expected[0]
    with pytest.raises((ValueError, OverflowError)):
        gpt2.encode(texts[0], max_threads=0)


@pytest.mark.filterwarnings("ignore:.*fork.*:DeprecationWarning")
@pytest.mark.parametrize("method", ["spawn", "fork"])
def test_multiprocessing_workers_start_serial(method):
    if method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"no {method}")
    with multiprocessing.get_context(method).Pool(1) as pool:
        assert pool.apply(tokenizers.get_max_threads) == 1
