import sys
import sysconfig
import threading

import pytest

import tokenizers


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


def test_set_max_threads_round_trips():
    default = tokenizers.get_max_threads()
    tokenizers.set_max_threads(3)
    try:
        assert tokenizers.get_max_threads() == 3
    finally:
        tokenizers.set_max_threads(0)
    assert tokenizers.get_max_threads() == default


# datasets, transformers and sglang set it to turn parallelism off.
@pytest.mark.parametrize("value", ["false", "0", "OFF"])
def test_tokenizers_parallelism_off_means_one_thread(monkeypatch, value):
    tokenizers.set_max_threads(3)
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", value)
    try:
        assert tokenizers.get_max_threads() == 1
    finally:
        tokenizers.set_max_threads(0)


# In 0.23 `true` kept threads on after a fork, where they can deadlock.
def test_tokenizers_parallelism_true_forces_nothing(monkeypatch):
    tokenizers.set_max_threads(1)
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")
    try:
        assert tokenizers.get_max_threads() == 1
    finally:
        tokenizers.set_max_threads(0)
