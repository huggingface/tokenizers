import base64
import functools
import gzip
import json
import os
import random
import time
from typing import Any, cast

import blobfile

import tiktoken
from tokenizers import Tokenizer


def format_byte_size(num_bytes: int) -> str:
    """Convert bytes to a human-readable format (KB, MB, GB)."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}", unit
        num_bytes /= 1024
    return f"{num_bytes:.2f} PB", unit


def benchmark_batch(documents: list[str]) -> None:
    num_threads = int(os.environ["RAYON_NUM_THREADS"])
    num_bytes = sum(map(len, map(str.encode, documents)))
    readable_size, unit = format_byte_size(num_bytes)
    print(f"num_threads: {num_threads}, data size: {readable_size}")
    enc = tiktoken.get_encoding("gpt2")
    enc.encode("warmup")

    start = time.perf_counter_ns()
    enc.encode_ordinary_batch(documents, num_threads=num_threads)
    end = time.perf_counter_ns()

    readable_size, unit = format_byte_size(num_bytes / (end - start) * 1e9)
    print(f"tiktoken \t{readable_size}  / s")

    import transformers

    hf_enc = Tokenizer.from_pretrained("gpt2")
    hf_enc.encode("warmup")

    start = time.perf_counter_ns()
    hf_enc.encode_batch(documents)
    end = time.perf_counter_ns()
    readable_size, unit = format_byte_size(num_bytes / (end - start) * 1e9)
    print(f"huggingface \t{readable_size} / s")


import os
import time
import tqdm
from datasets import load_dataset
import tiktoken


def test_on_xnli():
    dataset_xnli = load_dataset("facebook/xnli", "all_languages")

    # Varying the number of threads and length of input
    num_threads_list = [1, 4, 8, 16, 32]  # Example thread counts
    input_lengths = [10, 100, 1000, 10_000]  # Example input lengths

    documents = ["".join(item["premise"].values()) for item in dataset_xnli["train"]]
    for num_threads in num_threads_list:
        os.environ["RAYON_NUM_THREADS"] = str(num_threads)
        os.environ["TOKENIZER_PARALLELISM"] = str(num_threads)
        os.environ["RAYON_RS_NUM_THREADS"] = str(num_threads)
        for length in input_lengths:
            if length == 100_000 and num_threads == 1:
                break
            benchmark_batch(documents[:length])


# Call the function to run the benchmark
test_on_xnli()
