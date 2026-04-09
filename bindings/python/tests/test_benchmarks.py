"""
Benchmark suite for Python tokenizer bindings.

Measures the overhead of the Python ↔ Rust FFI layer by exercising the same
operations benchmarked on the Rust side.  Run with:

    pytest tests/test_benchmarks.py --benchmark-columns=mean,stddev,rounds -v

Requires: pytest-benchmark, tokenizers (built with maturin develop --release)
"""

import asyncio
import concurrent.futures
from pathlib import Path

import pytest

pytest.importorskip("pytest_benchmark")

from tokenizers import Tokenizer, AddedToken
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.trainers import BpeTrainer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "tokenizers" / "data"


@pytest.fixture(scope="module")
def big_text():
    return (DATA_DIR / "big.txt").read_text()


@pytest.fixture(scope="module")
def lines(big_text):
    return big_text.splitlines()


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    bpe = BPE.from_file(
        str(DATA_DIR / "gpt2-vocab.json"),
        str(DATA_DIR / "gpt2-merges.txt"),
    )
    tok = Tokenizer(bpe)
    tok.pre_tokenizer = ByteLevelPreTokenizer()
    return tok


@pytest.fixture(scope="module")
def llama3_tokenizer():
    return Tokenizer.from_file(str(DATA_DIR / "llama-3-tokenizer.json"))


@pytest.fixture(scope="module")
def roberta_tokenizer():
    return Tokenizer.from_file(str(DATA_DIR / "roberta.json"))


@pytest.fixture(scope="module")
def albert_tokenizer():
    return Tokenizer.from_file(str(DATA_DIR / "albert-base-v1-tokenizer.json"))


# ---------------------------------------------------------------------------
# Encoding benchmarks — GPT-2
# ---------------------------------------------------------------------------


class TestBPEGPT2:
    def test_encode(self, benchmark, gpt2_tokenizer, lines):
        def run():
            for line in lines:
                gpt2_tokenizer.encode(line)

        benchmark(run)

    def test_encode_batch(self, benchmark, gpt2_tokenizer, lines):
        benchmark(gpt2_tokenizer.encode_batch, lines)

    def test_encode_batch_multithreaded(self, benchmark, gpt2_tokenizer, lines):
        """encode_batch with multiple OS threads via concurrent.futures."""
        n_workers = 4
        chunk_size = len(lines) // n_workers

        def run():
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = []
                for i in range(n_workers):
                    chunk = lines[i * chunk_size : (i + 1) * chunk_size]
                    futures.append(pool.submit(gpt2_tokenizer.encode_batch, chunk))
                for f in futures:
                    f.result()

        benchmark(run)


# ---------------------------------------------------------------------------
# Encoding benchmarks — Llama-3
# ---------------------------------------------------------------------------


class TestLlama3:
    def test_encode(self, benchmark, llama3_tokenizer, lines):
        def run():
            for line in lines:
                llama3_tokenizer.encode(line)

        benchmark(run)

    def test_encode_batch(self, benchmark, llama3_tokenizer, lines):
        benchmark(llama3_tokenizer.encode_batch, lines)

    def test_encode_fast(self, benchmark, llama3_tokenizer, lines):
        """encode without offset tracking."""
        benchmark(llama3_tokenizer.encode_batch_fast, lines)

    def test_encode_batch_multithreaded(self, benchmark, llama3_tokenizer, lines):
        n_workers = 4
        chunk_size = len(lines) // n_workers

        def run():
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = []
                for i in range(n_workers):
                    chunk = lines[i * chunk_size : (i + 1) * chunk_size]
                    futures.append(pool.submit(llama3_tokenizer.encode_batch, chunk))
                for f in futures:
                    f.result()

        benchmark(run)

    def test_decode_batch(self, benchmark, llama3_tokenizer, lines):
        # Pre-encode to get token IDs for decoding
        encoded = llama3_tokenizer.encode_batch(lines[:1000])
        ids_list = [enc.ids for enc in encoded]
        benchmark(llama3_tokenizer.decode_batch, ids_list)


# ---------------------------------------------------------------------------
# Async encoding benchmarks
# ---------------------------------------------------------------------------


class TestAsync:
    def test_async_encode_batch(self, benchmark, llama3_tokenizer, lines):
        async def run():
            return await llama3_tokenizer.async_encode_batch(lines)

        benchmark(lambda: asyncio.run(run()))

    def test_async_encode_batch_fast(self, benchmark, llama3_tokenizer, lines):
        async def run():
            return await llama3_tokenizer.async_encode_batch_fast(lines)

        benchmark(lambda: asyncio.run(run()))


# ---------------------------------------------------------------------------
# Serialization benchmarks
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_from_file_roberta(self, benchmark):
        benchmark(Tokenizer.from_file, str(DATA_DIR / "roberta.json"))

    def test_from_file_llama3(self, benchmark):
        benchmark(Tokenizer.from_file, str(DATA_DIR / "llama-3-tokenizer.json"))

    def test_from_file_albert(self, benchmark):
        benchmark(Tokenizer.from_file, str(DATA_DIR / "albert-base-v1-tokenizer.json"))

    def test_to_str_llama3(self, benchmark, llama3_tokenizer):
        benchmark(llama3_tokenizer.to_str)

    def test_from_str_llama3(self, benchmark, llama3_tokenizer):
        json_str = llama3_tokenizer.to_str()
        benchmark(Tokenizer.from_str, json_str)


# ---------------------------------------------------------------------------
# Training benchmark
# ---------------------------------------------------------------------------


class TestTraining:
    def test_train_bpe_small(self, benchmark):
        def run():
            tok = Tokenizer(BPE())
            tok.pre_tokenizer = ByteLevelPreTokenizer()
            trainer = BpeTrainer(vocab_size=1000, show_progress=False)
            tok.train([str(DATA_DIR / "small.txt")], trainer)

        benchmark(run)
