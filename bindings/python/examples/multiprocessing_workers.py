"""
Encode in worker processes, with the tokenizer sent to each one

    python examples/multiprocessing_workers.py [path/to/tokenizer.json]

`multiprocessing` moves objects between processes by pickling them: the tokenizer travels to a
worker as a pickle, and the encodings travel back the same way. PyTorch's `DataLoader` and
`datasets.map(num_proc=...)` work like this too.

The tokenizer is pickled again with every task it is sent with, so give each worker one big chunk
of texts rather than many small tasks. Try one process first: `encode_batch` already spreads a large
batch over every core, and every worker process builds a thread pool of its own on top of that.

Defaults to the GPT-2 fixture `make test` fetches into `data/`.
"""

import multiprocessing
import sys
from pathlib import Path

from tokenizers import Tokenizer

path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent.parent / "data" / "gpt2.json")
texts = ["Hello, I'm a", "The weather today is quite a bit warmer than expected"] * 100
workers = 4


def encode(tokenizer, texts):
    """Runs in a worker process, on one chunk of the texts."""
    return tokenizer.encode_batch(texts)


# Each worker imports this file to find `encode`, so everything below has to run in the parent
# only. Without this guard, a worker would start workers of its own.
if __name__ == "__main__":
    tokenizer = Tokenizer.from_file(path)
    size = len(texts) // workers
    chunks = [texts[start : start + size] for start in range(0, len(texts), size)]

    # "spawn" starts every worker as a fresh interpreter, so the tokenizer has to travel as a
    # pickle. "fork", the default on Linux, hands the worker a copy of this process instead.
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        encoded = pool.starmap(encode, [(tokenizer, chunk) for chunk in chunks])

    for worker, encodings in enumerate(encoded):
        print(f"worker {worker}: {len(encodings)} texts, {sum(len(e) for e in encodings)} tokens")

    from_workers = [encoding.ids for chunk in encoded for encoding in chunk]
    print("same ids as encoding here:", from_workers == [tokenizer.encode(text).ids for text in texts])
