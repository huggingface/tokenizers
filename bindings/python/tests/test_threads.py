import sys
import sysconfig
import threading

import pytest


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


# `sys._is_gil_enabled` exists on every build from 3.13 on, so only the config var tells the two apart.
@pytest.mark.skipif(not sysconfig.get_config_var("Py_GIL_DISABLED"), reason="only meaningful on a free-threaded build")
def test_gil_disabled():
    assert sys._is_gil_enabled() is False  # ty: ignore[missing-attribute]
