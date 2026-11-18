import sys
import sysconfig
import threading

import numpy as np
import pytest
import torch

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


def test_read_buffer_while_decoding_it(gpt2):
    encoding = gpt2.encode(GPT2.read_text()[:4000])
    ids = encoding.ids_buffer

    def decode():
        for _ in range(100):
            gpt2.decode(ids)

    decoder = threading.Thread(target=decode)
    decoder.start()
    while decoder.is_alive():
        encoding.ids_buffer
    decoder.join()


def test_torch_reads_buffer_while_decoding_it(gpt2):
    encoding = gpt2.encode(GPT2.read_text()[:4000])
    ids = encoding.ids_buffer
    tensors = []

    def read():
        for _ in range(100):
            tensors.append(torch.tensor(ids))

    readers = [threading.Thread(target=read) for _ in range(4)]
    for reader in readers:
        reader.start()
    while any(reader.is_alive() for reader in readers):
        gpt2.decode(ids)
    for reader in readers:
        reader.join()

    assert len(tensors) == 400
    assert all(tensor.tolist() == encoding.ids for tensor in tensors)


def test_threads_export_the_same_memory(gpt2):
    encoding = gpt2.encode(GPT2.read_text()[:4000])
    start = threading.Barrier(8)
    arrays = []

    def export():
        start.wait()
        arrays.append(np.asarray(encoding.type_ids_buffer))

    threads = [threading.Thread(target=export) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(arrays) == 8
    assert all(np.shares_memory(array, arrays[0]) for array in arrays)


# `sys._is_gil_enabled` exists on every build from 3.13 on, so only the config var tells the two apart.
@pytest.mark.skipif(not sysconfig.get_config_var("Py_GIL_DISABLED"), reason="only meaningful on a free-threaded build")
def test_gil_disabled():
    assert sys._is_gil_enabled() is False  # ty: ignore[missing-attribute]
