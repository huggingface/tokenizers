import sys
import sysconfig
import threading

import pytest

from conftest import GPT2
from tokenizers import Padding


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


def test_set_padding_while_encoding(gpt2):
    texts = [GPT2.read_text()[:4000]] * 50
    failures = []

    def encode_repeatedly():
        try:
            for _ in range(10):
                gpt2.encode_batch(texts)
        except Exception as e:
            failures.append(e)

    encoder = threading.Thread(target=encode_repeatedly)
    encoder.start()
    while encoder.is_alive():
        gpt2.padding = Padding(pad_id=50256)
        gpt2.padding = None
    encoder.join()

    assert failures == []


# `sys._is_gil_enabled` exists on every build from 3.13 on, so only the config var tells the two apart.
@pytest.mark.skipif(not sysconfig.get_config_var("Py_GIL_DISABLED"), reason="only meaningful on a free-threaded build")
def test_gil_disabled():
    assert sys._is_gil_enabled() is False  # ty: ignore[unresolved-attribute]
