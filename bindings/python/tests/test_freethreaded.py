"""Concurrency stress tests for free-threaded Python (3.14t).

Pairs with docs/free-threading-audit.md. Designed to catch:
  - data races on RwLock-guarded component state,
  - RwLock poisoning from a panicking setter,
  - segfaults / SystemErrors from mismatched lifetimes.

These tests run on regular CPython too — under the GIL they're a
no-op for race detection, but they verify the non-racey behavior is
unchanged.
"""

import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFD, Sequence as NormalizerSequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import ByteLevel
from tokenizers.trainers import BpeTrainer


def _is_free_threaded() -> bool:
    is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
    return callable(is_gil_enabled) and not is_gil_enabled()


pytestmark = pytest.mark.timeout(60)  # any of these hanging means a deadlock


def _make_tokenizer() -> Tokenizer:
    tok = Tokenizer(BPE())
    tok.pre_tokenizer = Whitespace()
    tok.normalizer = NormalizerSequence([NFD(), Lowercase()])
    return tok


class TestEncodeUnderConcurrentSetters:
    """N encoders + M setters racing on the same Tokenizer.

    Audit references: §1 (single-field setter), §2 (top-level swap),
    §6 (encode read guard).
    """

    def test_encode_while_swapping_post_processor(self):
        tok = _make_tokenizer()
        stop = threading.Event()
        encoded_count = 0
        encode_lock = threading.Lock()

        def encoder():
            nonlocal encoded_count
            local_count = 0
            while not stop.is_set():
                enc = tok.encode("the quick brown fox jumps over the lazy dog")
                # Tokens list must always be well-formed (no torn read).
                assert isinstance(enc.tokens, list)
                assert all(isinstance(t, str) for t in enc.tokens)
                local_count += 1
            with encode_lock:
                encoded_count += local_count

        def setter():
            while not stop.is_set():
                tok.post_processor = ByteLevel(trim_offsets=True)

        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(encoder) for _ in range(6)]
            futures += [ex.submit(setter) for _ in range(2)]
            threading.Event().wait(2.0)  # let them race for 2 seconds
            stop.set()
            for f in futures:
                f.result()  # surface any exception

        assert encoded_count > 0, "encoders should make progress"

    def test_encode_while_mutating_trainer_fields(self):
        """Audit §5: trainer field mutation should not race with encode.

        train() is not called here — we just verify that mutating trainer
        fields doesn't poison locks observed by other operations.
        """
        tok = _make_tokenizer()
        trainer = BpeTrainer(vocab_size=1000)
        stop = threading.Event()

        def encoder():
            while not stop.is_set():
                tok.encode("hello world")

        def trainer_mutator():
            n = 0
            while not stop.is_set():
                trainer.vocab_size = 1000 + (n % 4096)
                trainer.min_frequency = n % 5
                n += 1

        with ThreadPoolExecutor(max_workers=6) as ex:
            futures = [ex.submit(encoder) for _ in range(4)]
            futures += [ex.submit(trainer_mutator) for _ in range(2)]
            threading.Event().wait(1.5)
            stop.set()
            for f in futures:
                f.result()

        # Final state must still be readable — a poisoned RwLock would
        # raise here.
        _ = trainer.vocab_size

    def test_concurrent_setters_no_lock_poisoning(self):
        """Audit §1: concurrent setters serialize through RwLock.

        If any setter panics inside the guarded scope, the lock is poisoned
        and the next reader raises. This test asserts neither happens
        under heavy contention.
        """
        tok = _make_tokenizer()
        stop = threading.Event()

        def setter_a():
            while not stop.is_set():
                tok.pre_tokenizer = Whitespace()

        def setter_b():
            while not stop.is_set():
                tok.normalizer = NFD()

        def reader():
            while not stop.is_set():
                _ = tok.pre_tokenizer
                _ = tok.normalizer

        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(setter_a) for _ in range(2)]
            futures += [ex.submit(setter_b) for _ in range(2)]
            futures += [ex.submit(reader) for _ in range(4)]
            threading.Event().wait(1.5)
            stop.set()
            for f in futures:
                f.result()

        # Final assignments succeed → locks are healthy.
        tok.pre_tokenizer = Whitespace()
        tok.normalizer = NFD()


@pytest.mark.skipif(not _is_free_threaded(), reason="3.14t-only check")
class TestFreeThreadedSpecific:
    """Asserts the 3.14t-specific properties: GIL truly off + module
    declares Py_MOD_GIL_NOT_USED."""

    def test_gil_actually_disabled_on_import(self):
        """If the wheel were misconfigured with gil_used=true, importing
        tokenizers would silently re-enable the GIL on 3.14t."""
        import tokenizers  # noqa: F401  (re-import is a no-op)

        assert sys._is_gil_enabled() is False, (
            "tokenizers re-enabled the GIL on free-threaded Python — wheel "
            "was built without gil_used=false"
        )
